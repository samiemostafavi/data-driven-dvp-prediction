import sys
import functools
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
from pathlib import Path
import os
import polars as pl

import qsimpy
from arrivals import HeavyTail

p = Path(__file__).parents[0]
results_path = str(p) + '/results2/'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_run_graph(params):
    # params = {
    #   'arrivals_number' : 1500000,
    #   'run_number' : 0,
    #   'arrival_seed' : 100234,
    #   'service_seed' : 120034,
    #   'gpd_concentration' : 0.4,
    #   'until': int(1000000),
    #   'report_state' : 0.1,
    # }

    # arrival function: Uniform
    arrival_rate = 0.091
    rng_arrival = np.random.default_rng(params['arrival_seed'])
    arrival = functools.partial(rng_arrival.uniform, 1.00/arrival_rate, 1.00/arrival_rate)

    # Gamma distribution
    ht = HeavyTail(
        n = params['arrivals_number'],
        seed = params['service_seed'],
        gamma_concentration = 5,
        gamma_rate = 0.5,
        gpd_concentration = params['gpd_concentration'], #0.4, 0.3, 0.2, 0.1, 0.01
        threshold_qnt = 0.8,
        dtype = np.float32,
    )
    # mean is 10.807

    # get the function
    service = ht.get_rnd_heavy_tail

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    env = qsimpy.Environment(name='0')

    # report timesteps
    def report_state(time_step):
        yield env.timeout(time_step)
        print(f"{params['run_number']}: Simulation progress {100.0*float(env.now)/float(params['until'])}% done")

    for step in np.arange(0, params['until'], params['until']*params['report_state'], dtype=int):
        env.process(report_state(step))

    # Create a source
    source = qsimpy.Source(
        name='start-node',
        env=env,
        arrival_dist=arrival,
        task_type='0',
    )

    # a queue
    queue = qsimpy.SimpleQueue(
        name='queue',
        env=env,
        service_dist=service,
        queue_limit=10,
    )

    # a sink: to capture both finished tasks and dropped tasks (compare PolarSink vs Sink)
    sink = qsimpy.PolarSink(
        name='sink',
        env=env,
        debug=False,
        batch_size = 10000,
    )

    # define postprocess function
    def process_time_in_service(df):
 
        df['end2end_delay'] = df['end_time']-df['start_time']
        df['service_delay'] = df['end_time']-df['service_time']
        df['queue_delay'] = df['service_time']-df['queue_time']

        df['time_in_service'] = df.apply(
                                lambda row: (row.start_time-row.last_service_time) if row.queue_is_busy else None,
                                axis=1,
                            ).astype('float64')

        del df['last_service_time'], df['queue_is_busy']

        return df

    sink.post_process_fn = process_time_in_service

    # Wire start-node, queue, end-node, and sink together
    source.out = queue
    queue.out = sink
    queue.drop = sink

    env.task_records = {
        'timestamps' : {
            source.name : {
                'task_generation':'start_time',
            },
            queue.name : {
                'task_reception':'queue_time',
                'service_start':'service_time',
                'service_end':'end_time',
            },
        },
        'attributes' : {
            source.name : {
                'task_generation' : {
                    queue.name : {
                        'queue_length':'queue_length',
                        'last_service_time':'last_service_time',
                        'is_busy':'queue_is_busy',
                    },
                },
            },
        },
    }

    # Run it
    start = time.time()
    env.run(until=params['until'])
    end = time.time()
    print("{0}: Run finished in {1} seconds".format(params['run_number'],end - start))

    print("{0}: Source generated {1} tasks".format(params['run_number'],source.get_attribute('tasks_generated')))
    print("{0}: Queue completed {1}, dropped {2}".format(
            params['run_number'],
            queue.get_attribute('tasks_completed'),
            queue.get_attribute('tasks_dropped'),
        )
    )
    print("{0}: Sink received {1} tasks".format(params['run_number'],sink.get_attribute('tasks_received')))

    start = time.time()

    # Process the collected data
    df = sink.received_tasks
    df_dropped = df.filter(pl.col('end_time') == -1)
    df_finished = df.filter(pl.col('end_time') >= 0)
    df = df_finished

    end = time.time()

    df.to_parquet(results_path + 'tailbench_{}.parquet'.format(params['run_number']))

    print("{0}: Data processing finished in {1} seconds".format(params['run_number'],end - start))



if __name__ == "__main__":

    sequential_runs = 1 # 10
    parallel_runs = 18 # 18
    for j in range(sequential_runs):

        processes = []
        for i in range(parallel_runs):
            params = {
                'arrivals_number' : 1500000,
                'run_number' : j*parallel_runs + i,
                'arrival_seed' : 100234+i*100101+j*10223,
                'service_seed' : 120034+i*200202+j*20111,
                'gpd_concentration' : 0.4,
                'until': int(1000000), # 10M timesteps takes 1000 seconds, generates 900k samples
                'report_state' : 0.1,
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