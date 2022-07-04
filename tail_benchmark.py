import functools
import numpy as np
import multiprocessing as mp
import time
from pathlib import Path
import os
import polars as pl
from dill.source import importable
import json

import qsimpy
from arrivals import HeavyTailGamma
from qsimpy.random import Deterministic, RandomProcess

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_run_graph(params):
    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = qsimpy.Model(name=f"tail benchmark #{params['run_number']}")

    # Create a source
    # arrival process deterministic
    arrival = Deterministic(
        rate = 0.095,
        seed = params['arrival_seed'],
        dtype = 'float64',
    )
    source = qsimpy.Source(
        name='start-node',
        arrival_rp=arrival,
        task_type='0',
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
    queue = qsimpy.SimpleQueue(
        name='queue',
        service_rp= service,
        queue_limit=10, #None
    )
    model.add_entity(queue)

    # Sink: to capture both finished tasks and dropped tasks (PolarSink to be faster)
    sink = qsimpy.PolarSink(
        name='sink',
        batch_size = 10000,
    )
    # define postprocess function: the name must be 'user_fn'

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df['end2end_delay'] = df['end_time']-df['start_time']
        df['service_delay'] = df['end_time']-df['service_time']
        df['queue_delay'] = df['service_time']-df['queue_time']
        # process time in service
        df['time_in_service'] = df.apply(
                                lambda row: (row.start_time-row.last_service_time) if row.queue_is_busy else None,
                                axis=1,
                            ).astype('float64')
        # process longer_delay_prob here for benchmark purposes
        df['longer_delay_prob'] = np.float64(1.00) - service.cdf(
            y = df['time_in_service'].to_numpy(),
        )
        df['longer_delay_prob'] = df['longer_delay_prob'].fillna(np.float64(0.00))
        del df['last_service_time'], df['queue_is_busy']
        return df

    # convert it to string and pass it to the sink function
    #user_fn_str = importable(user_fn, source=True)
    #sink.set_post_process_fn(fn_str=user_fn_str)
    #sink.set_post_process_fn

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    # Wire start-node, queue, end-node, and sink together
    source.out = queue.name
    queue.out = sink.name
    queue.drop = sink.name

    # Setup task records
    model.set_task_records({
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
    })

    modeljson = model.json()
    with open(params['records_path']+f"{params['run_number']}_model.json", 'w', encoding='utf-8') as f:
        f.write(modeljson)

    # prepare for run
    model.prepare_for_run(debug=False)

    # report timesteps
    def report_state(time_step):
        yield model.env.timeout(time_step)
        print(f"{params['run_number']}: Simulation progress {100.0*float(model.env.now)/float(params['until'])}% done")
    for step in np.arange(0, params['until'], params['until']*params['report_state'], dtype=int):
        model.env.process(report_state(step))

    # Run!
    start = time.time()
    model.env.run(until=params['until'])
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

    #print(df)

    end = time.time()
    
    df.to_parquet(params['records_path'] + f"{params['run_number']}_records.parquet")

    print("{0}: Data processing finished in {1} seconds".format(params['run_number'],end - start))



if __name__ == "__main__":

    # project folder setting
    p = Path(__file__).parents[0]
    project_path = str(p) + '/projects/tail_benchmark/'

    # simulation parameters
    bench_params = { # tail decays
        'p4':0.4, 
        'p3':0.3, 
        'p2':0.2, 
        'p1':0.1, 
        'pz':0.0001,
    }

    sequential_runs = 5 # 10
    parallel_runs = 18 # 18
    for j in range(sequential_runs):

        processes = []
        for i in range(parallel_runs):

            # parameter figure out
            keys = list(bench_params.keys())
            key_this_run = keys[j%len(keys)]

            # create and prepare the results directory
            results_path = project_path + key_this_run  + '_results/'
            records_path = results_path + 'records/'
            os.makedirs(records_path, exist_ok=True)

            params = {
                'records_path' : records_path,
                'arrivals_number' : 100000, #1.5M
                'run_number' : j*parallel_runs + i,
                'arrival_seed' : 100234+i*100101+j*10223,
                'service_seed' : 120034+i*200202+j*20111,
                'gpd_concentration' : bench_params[key_this_run], # tail decays
                'until': int(10000), # 10M timesteps takes 1000 seconds, generates 900k samples
                'report_state' : 0.1, # report when 10%, 20%, etc progress reaches
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