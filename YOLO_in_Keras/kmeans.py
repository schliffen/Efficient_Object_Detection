import numpy as np
import glob
import PIL
import matplotlib.pyplot as plt

anchors_save = '/home/ali/CLionProjects/object_detection/SqueezeDet/squeezedet/anchors/'

#
def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


class YOLO_Kmeans:

    def __init__(self, cluster_number, grid_w, grid_h, labelist, imgdir):
        self.cluster_number = cluster_number
        self.grid_w = grid_w
        self.grid_h = grid_h
        # self.filename = "2012_train.txt"
        self.filelist = labelist
        self.imgdir = imgdir

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open(anchors_save + "anchors_gball_" + str(self.grid_w) + ".txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%f,%f" % (data[i][0], data[i][1])
            else:
                x_y = ", %f,%f" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filelist, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txtlist2boxes(self):
        #
        # modify this file wrt the data structure
        dataSet = []
        #
        for item in self.filelist:
            dname = item[:-3].split('/')[-1]
            f = open( item, 'r')
            img = PIL.Image.open(self.imgdir + dname + 'jpg')
            # sometimes we need to read images to check image size etc
            w, h = img.size
            for line in f:
                infos = line.split(" ")[1:]
                # length = len(infos)
                # for i in range(1, length):
                # in the data with center and height width information
                width = float(infos[2]) * w/2 # int(infos[2]) - int(infos[0])
                height = float(infos[3][:-1]) * h  # int(infos[3][:-1]) - int(infos[1])
                dataSet.append([width, height])
            result = np.array(dataSet)
            f.close()
        return result

    def xml_to_boxes(self):
        # creating relative anchors
        import tensorflow as tf
        from lxml import etree

        dataSet =[]
        for example in self.filelist:
            with tf.gfile.GFile(example, 'rb') as fid:
                xml_str = fid.read()
                #
                xml = etree.fromstring(xml_str)
                data = recursive_parse_xml_to_dict(xml)['annotation']
                #
                for object in data['object']:
                    height = (float(object['bndbox']['ymax']) - float(object['bndbox']['ymin'])) /self.grid_h
                    width = (float(object['bndbox']['xmax']) - float(object['bndbox']['xmin'])) /self.grid_w
                    dataSet.append([width, height])

        return np.array(dataSet)

    def txt2clusters(self):
        # all_boxes = self.txtlist2boxes()
        all_boxes = self.xml_to_boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 5
    grid_w = 13
    grid_h = 13
    data_dir = '/home/ali/data/Golf_Ball_Data/'
    filelist = glob.glob(data_dir + 'p1_label/' + '*.xml')

    kmeans = YOLO_Kmeans(cluster_number, grid_w, grid_h, filelist, data_dir + 'p1/')
    # modify this part according to the dataset
    # kmeans.xml_to_boxes()
    # kmeans.txtlist2boxes()
    kmeans.txt2clusters()
