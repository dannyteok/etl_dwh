from twisted.web import server, resource
from twisted.web.static import File
from twisted.internet import reactor, endpoints
import worker
import jsonschema
from SOMTrainer.validation.post_message_validation import post_schema
import Queue
import os

input_queue = Queue.Queue()
processing_queue = Queue.Queue()
output_queue = Queue.Queue()
job_worker = worker.WorkerThread(input_queue=input_queue,
                                 processing_queue=processing_queue,
                                 completed_queue=output_queue
                                 )
def validate_json(configuration_json, json_schema):
    try:
        jsonschema.validate(configuration_json, json_schema)
        return True
    except jsonschema.ValidationError as error:
        print (error)
        return False


class PrintJsonPage(resource.Resource):
    isLeaf = True

    def render_POST(self, request):
        print request.args

class PrintSOMREport(resource.Resource):

    isLeaf = True

    def render_GET(self, request):
        pass

class SOMTrainerPage(resource.Resource):
    isLeaf = True
    job_count = 0

    def request_to_configuration(self,request):
        configuration_json = {}
        configuration_json['doIOS'] = bool(request.args["doIOS"][0])
        configuration_json['doAndroid'] = bool(request.args["doAndroid"][0])
        configuration_json['useMax'] = bool(request.args["useMax"][0])
        configuration_json['useMad'] = bool(request.args["useMad"][0])
        configuration_json['useMedian'] = bool(request.args["useMedian"][0])
        configuration_json['pcaComponentsCount'] = int(request.args["pcaComponentsCount"][0])
        configuration_json['somSize'] = int(request.args["somSize"][0])
        configuration_json['precision'] = int(request.args["precision"][0])
        configuration_json['clusters'] = int(request.args["clusters"][0])
        configuration_json['iterations'] = int(request.args["iterations"][0])
        configuration_json['maxIterations'] = int(request.args["maxIterations"][0])
        configuration_json['nodeScale'] = int(request.args["nodeScale"][0])
        configuration_json['edgeLength'] = float(request.args["edgeLength"][0])
        configuration_json['model'] = request.args["model"][0]
        configuration_json['floorplan'] = request.args["floorplan"][0]
        configuration_json['floor'] = request.args["floor"][0]
        configuration_json['site'] = request.args["site"][0]
        configuration_json['owner'] = request.args["owner"][0]
        configuration_json['job'] = request.args["job"][0]
        configuration_json['scenario'] = request.args["scenario"][0]
        configuration_json['beacons'] = request.args['beacons'][0]
        configuration_json['blueprint'] = request.args['blueprint'][0]
        configuration_json['fingeprints'] = request.args['fingeprints'][0]
        print (configuration_json)
        if validate_json(configuration_json,json_schema=post_schema):
            return configuration_json
        else:
            raise "Post Information Error"


    def TrainingAPI(self, request):

        print "training started queued jobs: {}, processing jobs: {}".format(input_queue.qsize(), processing_queue.qsize())
        configuration_json = self.request_to_configuration(request)
        input_queue.put(configuration_json)

    def render_POST(self, request):
        self.TrainingAPI(request=request)
        return "ok"




def start_server(port="8080"):
    job_worker.setDaemon(True)
    job_worker.start()
    root = File(os.path.join(os.getcwd(),'SOMTrainer/html'))
    root.putChild("trainer", SOMTrainerPage())
    root.putChild("receiver", PrintJsonPage())
    factory = server.Site(root)
    endpoints.serverFromString(reactor, "tcp:{}".format(port)).listen(factory)
    reactor.run()
    job_worker.join()
