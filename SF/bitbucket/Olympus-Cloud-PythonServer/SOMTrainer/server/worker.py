from __future__ import print_function
from training.training_manager import JobManager
import threading
import time

class WorkerThread(threading.Thread):
    def __init__(self, input_queue, completed_queue, processing_queue):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.completed_queue = completed_queue
        self.processing_queue = processing_queue
        self.job_manager = JobManager()

    def run(self):
        while True:
            configuration_json = self.input_queue.get()


            self.processing_queue.put(configuration_json['job'])
            print("job {} : starting training".format(configuration_json['job']))

            time.sleep(5)
            # task_done = self.job_manager.full_training(
            #         configuration_json=configuration_json,
            #     )
            self.job_manager.post_json_to_server(configuration_json)
            self.input_queue.task_done()
            print("job {} : completed training".format(configuration_json['job']))
            if self.processing_queue.qsize() > 0:
                self.processing_queue.get()
                self.processing_queue.task_done()

            self.completed_queue.put(configuration_json['job'])
