import sys
import functools
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
from pathlib import Path
import os

import qsimpy
from arrivals import HeavyTail

p = Path(__file__).parents[0]
results_path = str(p) + '/results/'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_run_graph(params):
    # params = {
    #   'run_number' : 0,
    #   'arrival_seed' : 100234,
    #   'service_seed' : 120034,
    #   'gpd_concentration' : 0.4,
    # }

    # arrival function: Uniform
    arrival_rate = 0.091
    rng_arrival = np.random.default_rng(params['arrival_seed'])
    arrival = functools.partial(rng_arrival.uniform, 1.00/arrival_rate, 1.00/arrival_rate)

    # Gamma distribution
    ht = HeavyTail(
        n = 1000000,
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

    # a sink: to capture both finished tasks and dropped tasks
    sink = qsimpy.Sink(
        name='sink',
        env=env,
        debug=False,
    )

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
    df = pd.DataFrame(sink.received_tasks)

    df_dropped = df[df.end_time==-1]
    df_finished = df[df.end_time>=0]
    df = df_finished

    df['end2end_delay'] = df['end_time']-df['start_time']
    df['service_delay'] = df['end_time']-df['service_time']
    df['queue_delay'] = df['service_time']-df['queue_time']
    df['time_in_service'] = df.apply(
                                lambda row: (row.start_time-row.last_service_time) if row.queue_is_busy else None,
                                axis=1,
                            )

    df.drop([
            'end_time',
            'start_time',
            'last_service_time',
            'queue_is_busy',
            'service_time',
            'queue_time',
        ], 
        axis = 1,
        inplace = True,
    )

    end = time.time()

    df.to_parquet(results_path + 'tailbench_{}.parquet'.format(params['run_number']))

    print("{0}: Data processing finished in {1} seconds".format(params['run_number'],end - start))




if __name__ == "__main__":

    sequential_runs = 1 # 11
    parallel_runs = 4 # 18
    for j in range(sequential_runs):

        processes = []
        for i in range(parallel_runs):
            params = {
                'run_number' : j*parallel_runs + i,
                'arrival_seed' : 100234+i*100101+j*10223,
                'service_seed' : 120034+i*200202+j*20111,
                'until': 1.1*1000000,
                'gpd_concentration' : 0.4,
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