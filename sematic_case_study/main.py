                                # REGION_TO_BE_EDITED_TRADITIONAL
import vid_pipeline
import camera
import neural_network
from utime import sleep
import plc_without_protocol as plc
import math
import uarray
import ustruct
import npufs as sdcard
import ujson
import utils                    
                                # REGION_TO_BE_EDITED_TEMPLATE
NUMBER_OF_NETWORK_OUTPUTS = 3 #number of outputs from your neural networt 
TARGER_RESOLUTION_WIDTH = 244 #image input width for the neural network
TARGER_RESOLUTION_HEIGHT = 244 #image input height for the neural network
NORM_MEAN_VALUE = 0 #mean value used for processing image

THRESHOLD = 0.5                 # END_REGION_TO_BE_EDITED_TEMPLATE
receivedFromPlc = {
    "inference":{
        "value":None,           # REGION_TO_BE_EDITED_TEMPLATE
        "type":"Int",
        "address":"%QW2"        # LINE_TO_BE_EDITED_SEMANTIC        # END_REGION_TO_BE_EDITED_TEMPLATE
    },
    "trigger":{
        "value":None,           # REGION_TO_BE_EDITED_TEMPLATE
        "type":"Bool",
        "address":"%Q4.0"       # LINE_TO_BE_EDITED_SEMANTIC        # END_REGION_TO_BE_EDITED_TEMPLATE
    }
}
sendToPlc = {
    "newDataAvailable":{
        "value":None,           # REGION_TO_BE_EDITED_TEMPLATE
        "type":"Bool",
        "address":"%I4.0"       # LINE_TO_BE_EDITED_SEMANTIC        # END_REGION_TO_BE_EDITED_TEMPLATE
    },
    "active":{
        "value":None,           # REGION_TO_BE_EDITED_TEMPLATE
        "type":"Bool",
        "address":"%I4.1"       # LINE_TO_BE_EDITED_SEMANTIC        # END_REGION_TO_BE_EDITED_TEMPLATE
    },
    "appStatus":{
        "value":None,           # REGION_TO_BE_EDITED_TEMPLATE
        "type":"Byte",
        "address":"%IB5"        # LINE_TO_BE_EDITED_SEMANTIC        # END_REGION_TO_BE_EDITED_TEMPLATE
    },
    "results":{
        "value":None,           # REGION_TO_BE_EDITED_TEMPLATE
        "type":"Array",
        "subtype":("Int", "Real"),
        "length":"24",
        "address":"%I6.0"       # LINE_TO_BE_EDITED_SEMANTIC        # END_REGION_TO_BE_EDITED_TEMPLATE
    }
} #END_REGION TO_BE_ADAPTED
#REGION INITIALIZE
print("START")

#init processimage and PLC object
output_buffer=bytearray(254)
appStatus = bytearray(1)
appStatusCode = 0 # startup
oldAppStatusCode = 0
active = 0
results = []
emptyResults = []
sendData = False
sendingDone = False
newDataAvailable = False
pathSdcardRoot = '/media/mmcsd-0-0/'
writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Start \n")
try:
    configs = get_config()
except:
    print("Reading configuration failed!")
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Reading configuration failed!\n")
    appStatusCode = 128 
try:
    for keys, values in receivedFromPlc.items():
        print("readReceive")
        writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "readReceive\n")
    for keys, values in sendToPlc.items():
        print("readSend")
        writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "readSend\n")
except:
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Reading UDTs failed!\n")
    print("Reading UDTs failed!")
    appStatusCode = 129
try:
    plc = plc.plc()
except:
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Initializing PLC communication failed!\n")
    appStatusCode = 130
camera_id=str(configs["camera_driver"])        
sleep(1)   
print("cameraid ", camera_id)
try:
    if camera_id == "GIGE_VISION":
        pfd_info = "UYVY"
    else:
        pfd_info = "YUYV"
    print("Initializing camera...")
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Initializing camera...\n")
    cam_info = camera.camera(camera_id = camera_id, resolution=(int(configs["frame_width"]), int(configs["frame_height"])),pixel_format_description=pfd_info)
   
    print("camera ok")
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "camera ok\n")
except:
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Initializing camera failed!\n")
    print("Initializing camera failed!")
    appStatusCode = 131
try:
    vid_pipeline.init(camera=cam_info,
        target_resolution=(TARGER_RESOLUTION_WIDTH, TARGER_RESOLUTION_HEIGHT),
        target_format="RGB",
        target_normalization=(NORM_MEAN_VALUE, NORM_SCALE_VALUE),
        target_output="float",
        shaves_scaling=int(
            configs["models"][0]["shaves_scaling"]),
        shaves_conversion=int(
            configs["models"][0]["shaves_conversion"]))
except:
    print("Initializing video pipeline failed!")
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Initializing video pipeline failed!\n")
    appStatusCode = 132
try:
    pathNN = pathSdcardRoot + 'res/'+str(configs["models"][0]["file"])
    netFile = readFromFile(pathNN, 'rb')
    net = neural_network.init(netFile)
except:
    print("Loading neural network failed!")
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Loading neural network failed!\n")
    appStatusCode = 133
try:
    sendToPlc["active"].update({"value":active})
    sendToPlc["appStatus"].update({"value":appStatusCode.to_bytes(1, 'big')})
    sendToPlc["newDataAvailable"].update({"value":newDataAvailable})
    sendToPlc["results"].update({"value":emptyResults})
    output_buffer = assembleUDTByteStream(sendToPlc)
    plc.write(bytes(output_buffer))
except:
    print("Writing to process image failed!")
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Writing to process image failed!\n")
    appStatusCode = 134

try:
    input_bytes = None
    while(type(input_bytes) == type(None)):
        input_bytes = plc.read()
        sleep(1)
    input_buffer = bytearray(input_bytes) 
    reassembleDataFromPLC(receivedFromPlc, input_buffer)
except:
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Reading from process image failed!\n")
    print("Reading from process image failed!")
    appStatusCode = 135
yoloParameters = YoloParams(params,side)
if (appStatusCode == 0):
    machineState = 1
    print("READY")
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "READY\n")
    appStatusCode = 1 # startup done, NPU is ready
else:
    print("AppStatus: ", appStatusCode)
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "AppStatus: ")
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', appStatusCode)
    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "\n")
#END_REGION INITIALIZE
while(True):
    sleep(1)   
    #REGION CONTINUOUS RECEIVING DATA
    input_bytes = plc.read()
    if(type(input_bytes) != type(None)):
        sleep(1)   
        input_buffer = bytearray(input_bytes) 
        reassembleDataFromPLC(receivedFromPlc,input_buffer)    
        #print value of each variable received from PLC
        for variable,properties in receivedFromPlc.items():
            print(variable + ": " + str(properties.get("value")))
    #END_REGION CONTINUOUS RECEIVING DATA    
    #REGION STATE MACHINE
    if (machineState == 1) and (receivedFromPlc["trigger"].get("value")==1): #waiting for trigger 
        print("machineState: TRIGGER RECEIVED")
        writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "machineState: TRIGGER RECEIVED\n")
        #Send active sign to PLC
        active = 1
        newDataAvailable = 0
        results = emptyResults
        appStatusCode = 2    
        sendData = True   
        if sendingDone == True:
            machineState = 2
            sendingDone = False
            sendData = False
    elif (machineState==2): #take image and run NN             
        print("machineState: INFERENCE")
        writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "machineState: INFERENCE\n")
        if receivedFromPlc["inference"].get("value") >= 1:
            try:
                writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Start streaming\n")
                vid_pipeline.start_streaming(1)
            except:
                print("Start streaming camera failed!")
                writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Start streaming camera failed!\n")
                appStatusCode = 136
        if(receivedFromPlc["inference"].get("value") >= 1): #image classification
            try:
                with vid_pipeline.read_processed() as frame:
                    unsortedResults = net.run(frame.data())
                    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "run net\n")
                    for ii in range(5):
                        index = unsortedResults[ii]
                        print("{0}".format(index))
                    if(receivedFromPlc["inference"].get("value")==1): #image classification
                        sortedResults = sortResults(unsortedResults)
                    elif(receivedFromPlc["inference"].get("value")==2): #object detection with YOLO postprocessing
                        sortedResults = parse_yolo_region(unsortedResults,yoloParameters, THRESHOLD)
                    elif(receivedFromPlc["inference"].get("value")==3): #object detection with mobilenet SSD postprocessing
                        sortedResults = processOutput(unsortedResults,int(configs["frame_width"]),int(configs["frame_height"]))       
                    print(sortedResults)
                    writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', sortedResults)
            except:
                print("Running neural network on processed image failed!")
                writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Running neural network on processed image failed!\n")
                appStatusCode = 138    
        else:
            sortedResults=[]        
        if (receivedFromPlc["inference"].get("value") >= 1):
            try:
                writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Start stop streaming\n")
                vid_pipeline.stop_streaming()
            except:
                print("Stopping video pipeline failed!")
                writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Stopping video pipeline failed!\n")
                appStatusCode = 139
        machineState = 3
    elif (machineState == 3): #writing results back to PLC
        print("machineState: WRITING RESULTS")
        writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "machineState: WRITING RESULTS\n")
        active=1
        #when newDataAvailable is 1, PLC knows new results are available
        newDataAvailable = 1
        results = sortedResults
        sendData = True        
        if (sendingDone == True):
            machineState = 4
            sendingDone = False
            sendData = False
        print("Done")
        writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Done\n")
    elif (machineState == 4) and (receivedFromPlc["trigger"].get("value") == 0):
        print("machineState: RESETTING VALUES")
        writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "machineState: RESETTING VALUES\n")
        #reset of active and newDataAvailable to get in inital state
        active=0
        newDataAvailable = 0
        results = sortedResults
        appStatusCode = 1
        sendData = True
        if (sendingDone == True):
            machineState = 1
            print("READY")
            writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "READY\n")
            sendingDone = False
            sendData = False
    #END_REGION STATE MACHINE
    #REGION CHECK APPSTATUS
    if (appStatusCode > 2) and (appStatusCode != oldAppStatusCode):
        sendData = True
        oldAppStatusCode = appStatusCode
    #END_REGION CHECK APPSTATUS
    #REGION SEND DATA
    #sending data only once per cycle.  When app error appears, programm directly jumps here
    if (sendData == True):
        print("Writing to PLC...")
        writeToFile(pathSdcardRoot + 'USERDATA/' + str('test.txt'), 'a+', "Writing to PLC...\n")
        sendToPlc["active"].update({"value":active})
        sendToPlc["appStatus"].update({"value":appStatusCode.to_bytes(1, 'big')})
        sendToPlc["newDataAvailable"].update({"value":newDataAvailable})
        sendToPlc["results"].update({"value":results})
        output_buffer = assembleUDTByteStream(sendToPlc)
        plc.write(bytes(output_buffer))
        sendData = False
        sendingDone = True
    #END_REGION SEND DATA    #END_REGION_TO_BE_EDITED_TRADITIONAL