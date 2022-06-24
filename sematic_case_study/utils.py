# neural netwrok speciic values and postprocessing
params = {'anchors': '10,14,23,27,37,58,81,82,135,169,344,319', 'axis': '1', 'classes': '1', 'coords': '4',
          'do_softmax': '0', 'end_axis': '3', 'mask': '3,4,5', 'num': '6'}
side = 13

NUMBER_OF_NETWORK_OUTPUTS = 3

class YoloParams:
    """
    This class and all following methods are necessary to postprocess the raw
    output from a YOLO neural network. It processes it to be in readable format
    and formats it so it can be send to the PLC.
    """

    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, params, threshold):
    orig_im_h, orig_im_w = 416, 416
    resized_image_h, resized_image_w = 416, 416
    objects = list()
    predictions = blob
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = math.exp(predictions[box_index + 2 * side_square])
                h_exp = math.exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))

    objects = sorted(objects, key=lambda obj: obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > 0.5:
                objects[j]['confidence'] = 0
    finalObjects = []
    for obj in objects:
        # Validation bbox of detected object
        if obj['xmax'] > 416 or obj['ymax'] > 416 or obj['xmin'] < 0 or obj['ymin'] < 0 or obj['confidence'] <= 0:
            continue
        else:
            finalObjects.append(
                (obj['confidence'], obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj['class_id']))

    if finalObjects == []:
        finalObjects.append((-1, -1, -1, -1, -1, -1))

    return finalObjects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def processOutput(output, width, height):
    """
    This method takes the raw output from a Tensorflow object detection API neural
    network (e.g. SSD mobilenet) and postprocesses it in a way to prepare it for
    sending it to the PLC
    """
    dic = []
    unsortedConf = []
    for k in range(0, len(output), 7):
        if output[k + 2] >= THRESHOLD:
            unsortedConf.append(output[k + 2])
    numOut = len(unsortedConf)
    if (numOut > 3):
        numOut = 3
    sorted = sortResults(unsortedConf, numOut, len(unsortedConf))
    print("Sorted indices " + str(sorted))
    for j, conf in sorted:
        k = j * 7
        x1 = int(output[k + 3] * width)
        y1 = int(output[k + 4] * height)
        x2 = int(output[k + 5] * width)
        y2 = int(output[k + 6] * height)
        index = int(output[k + 1])
        conf = output[k + 2]
        dic.append((conf, x1, y1, x2, y2, index))
    print("########################### processOutput " + str(dic))
    return dic


def sortResults(unsortedRes, numSortedRes=3, numNetOutputs=3):
    """
    This method brings the unsorted results from a neural network in the right order
    it returns a sorted array with 3 elements by default.
    Parameters:
    - unsortedRes: array of results returned from net.run() method
    """
    sortedRes = []
    for _ in range(numSortedRes):
        index = 0
        maxi = 0
        for ii in range(numNetOutputs):
            if maxi <= unsortedRes[ii] and (ii, unsortedRes[ii]) not in sortedRes:
                maxi = unsortedRes[ii]
                index = ii
        sortedRes.append((index, maxi))
    print("sortResults: Sorted the results")
    # log.info("sortResults: Sorted the results")
    return sortedRes


# function to read and write from sd card
def get_config():
    with sdcard.open('/media/mmcsd-0-0/res/npu_app.conf') as f:  # 'ftpServer/.....txt'
        return ujson.loads(f.read())


def readFromFile(filePath, mode):
    """
    This method reads file from sd card and returns its content
    Parameters:
    - filePath: path of the file on sd card
    - mode: mode of the read functionality ("r","w","a"...)
    """
    # log.info(filePath)
    try:
        with sdcard.open(filePath, mode) as fp:
            # log.info("readFromFile: File was opened")

            try:
                content = fp.read()
                # log.info("readFromFile: File was read")
                print("readFromFile: File was read")
            except:
                # log.error("readFromFile: Error when reading file")
                print("readFromFile: Error when reading file")
    except RuntimeError:
        # log.error("readFromFile: RuntimeError when opening file")
        print("readFromFile: RuntimeError when opening file")
    return content


def writeToFile(filePath, mode, content):
    """
    This method writes content to that file on the sd card
    Parameters:
    - filePath: path of the file on sd card
    - mode: mode of the read functionality ("r","w","a"...)
    - content: content which is written in the file
    """
    try:
        with sdcard.open(filePath, mode) as fp:
            # log.info("writeToFile: File was opened")
            try:
                fp.write(content)
                # log.info("writeToFile: File was written")
                print("writeToFile: File was written")
            except:
                print("writeToFile: Error when writing to file")
                # log.error("writeToFile: Error when writing to file")
    except RuntimeError:
        # log.error("writeToFile: RuntimeError when opening file")
        print("writeToFile: RuntimeError when opening file")


# functions to convert process image (bytes) into variable and back
def reassembleDataFromPLC(receivedFromPLC, input_buffer):
    """
    This method assembles a UDT structure from a bytestream send from PLC based on the variable receivedFromPlc
    It returns a dictionary with key-value pairs of a UDT structure from TIA Portal
    Parameters:
    - sendFromPLC: nested dictionary that describes the content of the process image between PLC and TM NPU
    - input_buffer: Bytestream send from PLC to be processed. Data is being extracted from this bytestream.
    """

    for variable in receivedFromPLC.values():

        if (variable.get("type") == "Bool"):
            position = variable.get("address").split(".")
            bytePosition = int(position[0][2:]) - 2
            bitPosition = position[1]
            variable.update({"value": hexToBin(input_buffer[bytePosition])[int(bitPosition)]})

        if (variable.get("type") == "DWord"):
            bytePosition = int(variable.get("address")[3:]) - 2
            variable.update({"value": bytes(input_buffer[bytePosition:bytePosition + 4])})

        if (variable.get("type") == "Int"):
            bytePosition = int(variable.get("address")[3:]) - 2
            variable.update({"value": hexToInt(input_buffer[bytePosition:bytePosition + 2])})

        if (variable.get("type") == "Array"):
            position = variable.get("address").split(".")
            bytePosition = int(position[0][2:]) - 2
            if (variable.get("subtype") == "Char"):
                ASCIIFileName = input_buffer[bytePosition:bytePosition + int(variable.get("length"))]
            file = ""
            for x in ASCIIFileName:
                if (x != 0):
                    file += str(chr(x))
                else:
                    continue
            variable.update({"value": file})

        if (variable.get("type") == "Date_And_Time"):
            position = variable.get("address").split(".")
            bytePosition = int(position[0][2:]) - 2
            dateAndTimeObj = hexToDateTime(bytes(input_buffer[bytePosition:bytePosition + 8]))
            dateAndTime = str(dateAndTimeObj.year) + "-" + str(dateAndTimeObj.month) + "-" + str(
                dateAndTimeObj.day)  # +"-"+str(dateAndTimeObj.hour)+":"+str(dateAndTimeObj.minute)+":"+str(dateAndTimeObj.second)
            variable.update({"value": dateAndTime})

        if (variable.get("type") == "Real"):
            bytePosition = int(variable.get("address")[3:]) - 2
            realValue = hexToReal(input_buffer[bytePosition:bytePosition + 4])
            variable.update({"value": realValue})


def assembleUDTByteStream(dictPLC):
    """
    This method assembles a bytestream to be sent back to PLC based on a UDT structure in TIA Portal
    It returns a bytestream to be send to PLC with data from dictPLC dictionary
    Parameters:
    - dictPLC: Dictionary containing structure and data to be send to PLC
    """
    assembly_buffer = bytearray(254)

    for variable in dictPLC.values():

        if (variable.get("type") == "Bool"):
            assembleBool(assembly_buffer, variable.get("address"), variable.get("value"))

        if (variable.get("type") == "Byte"):
            assembleByte(assembly_buffer, variable.get("address"), variable.get("value"))

        if (variable.get("type") == "Int"):
            assembleInt(assembly_buffer, variable.get("address"), variable.get("value"))

        if (variable.get("type") == "DWord"):
            assembleDWORD(assembly_buffer, variable.get("address"), variable.get("value"))

        if (variable.get("type") == "Real"):
            assembleReal(assembly_buffer, variable.get("address"), variable.get("value"))

        if (variable.get("type") == "Array"):
            position = variable.get("address").split(".")
            bytePosition = int(position[0][2:])

            bitPosition = 0
            oldSubtype = None

            if (isinstance(variable.get("subtype"), tuple)):
                for ii in range(len(variable.get(
                        "value"))):  # running through all array elements, but why are we using the length of "value" instad of number of elements "length"
                    valueIndex = 0
                    for subtype in variable.get("subtype"):  # for each subtype in an array element
                        if (subtype == "Int"):
                            assembleInt(assembly_buffer, str("%IW" + str(bytePosition)),
                                        int(variable.get("value")[ii][valueIndex]))
                            bytePosition = str(
                                int(bytePosition) + 2)  # increase bytePosition by two bytes which are used for each Int
                            bitPosition = 0  # Setting Bitposition back to 0

                        if (subtype == "Real"):
                            assembleReal(assembly_buffer, str("%ID" + str(bytePosition)),
                                         float(variable.get("value")[ii][valueIndex]))
                            bytePosition = str(
                                int(bytePosition) + 4)  # increase bytePosition by two bytes which are used for each Int
                            bitPosition = 0  # Setting Bitposition back to 0
                        oldSubtype = subtype
                        valueIndex = valueIndex + 1
            else:
                print("Error: Array subclass is not of type \"tuple\"")

            # if(variable.get("subtype")=="DWord"):
            #    for ii in range(len(variable.get("value"))):
            #        index, confidence = variable.get("value")[ii]

            #        val = intToHex(index)
            #        pos = ii*4+int(bytePosition)
            #        assembly_buffer[pos] = val[0]
            #        assembly_buffer[pos+1] = val[1]
            #        var = intToHex(int(confidence*100))
            #        assembly_buffer[pos+2] = var[0]
            #        assembly_buffer[pos+3] = var[1]

    # log.info(assembly_buffer)
    return assembly_buffer


def intToHex(inint):
    """
    This method converts integer to a hexadecimal number represented by 2 bytes with big endian.
    Parameters:
    - inint: integer value to be converted
    """
    return inint.to_bytes(2, 'big')


def hexToBin(inhex):
    """
    This method converts hexadecimal number to binary representation. It returns binary number as array.
    Parameters:
    - inhex: hexadecimal value to be converted
    """
    mant = inhex
    retbin = []
    for _ in range(8):
        retbin.append(int(math.fmod(mant, 2)))
        mant = math.floor((mant / 2))
    return retbin


def hexToInt(inhex):
    """
    This method converts hexadecimal number to integer representation. It returns an integer number.
    Parameters:
    - inhex: hexadecimal value to be converted
    """
    # log.info("Converting bytearray to int: " + str(inhex))
    convertedInt = int.from_bytes(inhex, 'big')
    # log.info("Converted integer: " + str(convertedInt))
    return convertedInt


def bcdToInt(inbcd):
    """
    This method converts BCD data type to integer
    Parameters:
    - inbcd: BCD value to be converted
    """
    intbcd = hexToBin(inbcd)
    bcdhigh = ((intbcd[7] * 2 + intbcd[6]) * 2 + intbcd[5]) * 2 + intbcd[4]
    bcdlow = ((intbcd[3] * 2 + intbcd[2]) * 2 + intbcd[1]) * 2 + intbcd[0]
    return bcdhigh * 10 + bcdlow


def hexToDateTime(inhexDT):
    """
    This method converts hexadecimal representation to date and time.
    It returns an object of a DT class.
    Parameters:
    - inhexDT: Hexadecimal value to be converted(8 bytes)
    """

    class DT:
        def __init__(self, inhex):
            year = bcdToInt(inhex[0])
            if year >= 90:
                self.year = 1900 + year
            else:
                self.year = 2000 + year
            self.month = bcdToInt(inhex[1])
            self.day = bcdToInt(inhex[2])
            self.hour = bcdToInt(inhex[3])
            self.minute = bcdToInt(inhex[4])
            self.second = bcdToInt(inhex[5])
            mSecHelper = hexToBin(inhex[7])
            self.mSecond = bcdToInt(inhex[6]) * 10 + (
                        ((mSecHelper[7] * 2 + mSecHelper[6]) * 2 + mSecHelper[5]) * 2 + mSecHelper[4])
            weekdayHelper = ((mSecHelper[3] * 2 + mSecHelper[2]) * 2 + mSecHelper[1]) * 2 + mSecHelper[0]
            if weekdayHelper == 1:
                self.dayOfWeek = "Sunday"
            elif weekdayHelper == 2:
                self.dayOfWeek = "Monday"
            elif weekdayHelper == 3:
                self.dayOfWeek = "Tuesday"
            elif weekdayHelper == 4:
                self.dayOfWeek = "Wednesday"
            elif weekdayHelper == 5:
                self.dayOfWeek = "Thursday"
            elif weekdayHelper == 6:
                self.dayOfWeek = "Friday"
            elif weekdayHelper == 7:
                self.dayOfWeek = "Saturday"

    return DT(inhexDT)


def hexToReal(inhex):
    """
    This method converts hexadecimal number to real representation. It returns a real number.
    Parameters:
    - inhex: hexadecimal value to be converted
    """
    intbuff = uarray.array('B')
    for _ in inhex:
        intbuff.append(_)
    intbuff.append(0x00)
    intbuff.append(0x00)
    intbuff.append(0x00)
    intbuff.append(0x00)
    return ustruct.unpack('>f', intbuff)[0]


def realToHex(inreal):
    """
    This method converts real to a hexadecimal number represented by 4 bytes.
    Parameters:
    - inint: integer value to be converted
    """
    retreal = bytearray(4)
    cint = 3
    for _ in bytearray(uarray.array('f', [inreal])):
        retreal[cint] = intToHex(_)[1]
        cint -= 1
    return retreal


def getList(dictToList):
    """
    This method is convertint dictionary data type to list(array) of keys
    It returns a list of keys from dictionary
    Parameters:
    - dict: dictionary to be converted
    """
    resultList = []
    for key in dictToList.keys():
        resultList.append(key)
    return resultList


def assembleByte(assembly_buffer, address, value):
    """
    This method is a helper function for assembleUDTBytestream
    """
    bytePosition = int(address[3:]) - 2  # shift of 2 because of firmware bytes
    assembly_buffer[bytePosition:bytePosition + 1] = value


def assembleInt(assembly_buffer, address, value):
    """
    This method is a helper function for assembleUDTBytestream
    """
    bytePosition = int(address[3:]) - 2  # shift of 2 because of firmware bytes
    assembly_buffer[bytePosition:bytePosition + 2] = intToHex(value)


def assembleDWORD(assembly_buffer, address, value):
    """
    This method is a helper function for assembleUDTBytestream
    """
    bytePosition = int(address[3:]) - 2  # shift of 2 because of firmware bytes
    assembly_buffer[bytePosition:bytePosition + 4] = value


def assembleReal(assembly_buffer, address, value):
    """
    This method is a helper function for assembleUDTBytestream
    """
    bytePosition = int(address[3:]) - 2  # shift of 2 because of firmware bytes
    assembly_buffer[bytePosition:bytePosition + 4] = realToHex(value)


def assembleBool(assembly_buffer, address, value):
    """
    This method is a helper function for assembleUDTBytestream
    """
    position = address.split(".")
    bytePosition = int(position[0][2:]) - 2  # shift of 2 because of firmware bytes
    bitPosition = position[1]
    manipulated_byte = assembly_buffer[bytePosition]
    mask = 2 ** int(bitPosition)  # masks which bit is affected in the byte
    if (value):  # sets the bit of the mask to 1
        manipulated_byte |= mask
    else:  # sets the bit of the mask to 0
        manipulated_byte &= ~mask
    assembly_buffer[bytePosition] = manipulated_byte


def bitIntoBytes(bitVal, bitPos, bytePos, Bytes):
    """
    This method puts bit value in the right position of a bytearray
    It returns a bytearray of 1 byte
    Parameters:
    - bitVal: bit value
    - bitPos: position on the byte where the bitVal should be placed
    - bytePos: position of a byte in bytearray
    . Bytes: byte array to be changed
    """
    binaryNum = hexToBin(Bytes[int(bytePos)])
    binaryNum[int(bitPos)] = bitVal
    ##log.info("Converted binary as tuple: " + binaryNum)
    binaryNumRev = binaryNum[::-1]
    # print(binaryNumRev)
    binaryStr = ""
    for j in binaryNumRev:
        binaryStr += str(j)
    ##log.info("Converted binary as tuple: " + binaryStr)
    decimal_representation = int(binaryStr, 2)
    # print(decimal_representation)
    Bytes[int(bytePos)] = decimal_representation
    return Bytes

