import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from qsimpy.core import Model, Sink
from qsimpy.gym import GymSink, GymSource
from qsimpy.random import RandomProcess
from arrivals import HeavyTailGamma
from qsimpy.simplequeue import SimpleQueue


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_run_graph(params):
    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"Gym benchmark #{params['run_number']}")

    # create the gym source
    source = GymSource(
        name="start-node",
        main_task_type="main",
        traffic_task_type="traffic",
        traffic_task_num=params["traffic_tasks"],
    )
    model.add_entity(source)

    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed = params['service_seed'],
        gamma_concentration = 5,
        gamma_rate = 0.5,
        gpd_concentration = params['gpd_concentration'],
        threshold_qnt = 0.8,
        dtype = 'float64',
        batch_size = params['arrivals_number'],
    )

    queue = SimpleQueue(
        name="queue",
        service_rp=service,
        queue_limit=None,  # 10, None
    )
    model.add_entity(queue)

    # create the sinks
    sink = GymSink(
        name="gym-sink",
        batch_size=10000,
    )

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df["end2end_delay"] = df["end_time"] - df["start_time"]
        df["service_delay"] = df["end_time"] - df["service_time"]
        df["queue_delay"] = df["service_time"] - df["queue_time"]
        return df

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    drop_sink = Sink(
        name="drop-sink",
    )
    model.add_entity(drop_sink)

    # make the connections
    source.out = queue.name
    queue.out = sink.name
    queue.drop = drop_sink.name
    sink.out = source.name
    # queue should not drop any task

    # Setup task records
    model.set_task_records(
        {
            "timestamps": {
                source.name: {
                    "task_generation": "start_time",
                },
                queue.name: {
                    "task_reception": "queue_time",
                    "service_start": "service_time",
                },
                sink.name: {
                    "task_reception": "end_time",
                },
            },
            "attributes": {
                source.name: {
                    "task_generation": {
                        queue.name: {
                            "queue_length": "queue_length",
                            "is_busy": "queue_is_busy",
                        },
                    },
                },
            },
        }
    )

    modeljson = model.json()
    with open(
        params["records_path"] + f"{params['run_number']}_model.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(modeljson)

    # prepare for run
    model.prepare_for_run(debug=False)

    # report timesteps
    def report_state(time_step):
        yield model.env.timeout(time_step)
        logger.info(
            f"{params['run_number']}:"
            + " Simulation progress"
            + f" {100.0*float(model.env.now)/float(params['until'])}% done"
        )

    for step in np.arange(
        0, params["until"], params["until"] * params["report_state"], dtype=int
    ):
        model.env.process(report_state(step))

    # Run!
    start = time.time()
    model.env.run(until=params["until"])
    end = time.time()
    logger.info(f"{params['run_number']}: Run finished in {end - start} seconds")

    logger.info(
        "{0}: Source generated {1} main tasks".format(
            params["run_number"], source.get_attribute("main_tasks_generated")
        )
    )
    logger.info(
        "{0}: Queue completed {1}, dropped {2}".format(
            params["run_number"],
            queue.get_attribute("tasks_completed"),
            queue.get_attribute("tasks_dropped"),
        )
    )
    logger.info(
        "{0}: Sink received {1} main tasks".format(
            params["run_number"], sink.get_attribute("tasks_received")
        )
    )

    start = time.time()

    # Process the collected data
    df = sink.received_tasks
    # df_dropped = df.filter(pl.col("end_time") == -1)
    df_finished = df.filter(pl.col("end_time") >= 0)
    df = df_finished

    # print(df)

    end = time.time()

    df.write_parquet(
        file=params["records_path"] + f"{params['run_number']}_records.parquet",
        compression="snappy",
    )

    logger.info(
        "{0}: Data processing finished in {1} seconds".format(
            params["run_number"], end - start
        )
    )


if __name__ == "__main__":

    # project folder setting
    p = Path(__file__).parents[0]
    project_path = str(p) + "/projects/qlen_benchmark/"

    # simulation parameters
    # bench_params = {str(n): n for n in range(15)}
    bench_params = {
        '15':15,
        '16':16,
        '17':17,
        '18':18,
        '19':19,
        '20':20,
        '21':21,
        '22':22,
        '23':23,
        '24':24,
        '25':25,
        '26':26,
        '27':27,
        '28':28,
        '29':29,
        '30':30,
    }

    sequential_runs = 4  # 5
    parallel_runs = 16  # 18
    for j in range(sequential_runs):

        processes = []
        for i in range(parallel_runs):  # range(parallel_runs):

            # parameter figure out
            keys = list(bench_params.keys())
            # remember to modify this line
            key_this_run = keys[i % len(keys)]

            # create and prepare the results directory
            results_path = project_path + key_this_run + "_results/"
            records_path = results_path + "records/"
            os.makedirs(records_path, exist_ok=True)

            until = int(10000000 * (bench_params[key_this_run] + 1))
            params = {
                "records_path": records_path,
                'arrivals_number' : int(until/10),
                "run_number": j * parallel_runs + i,
                "service_seed": 120034 + i * 200202 + j * 20111,
                'gpd_concentration' : 0.1,  # tail decay
                "traffic_tasks": bench_params[key_this_run],  # number of traffic tasks
                "until": until,  # 10M timesteps takes 1000 seconds, generates 900k samples
                "report_state": 0.05,  # report when 10%, 20%, etc progress reaches
            }

            p = mp.Process(target=create_run_graph, args=(params,))
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
                p.join()
                exit(0)
