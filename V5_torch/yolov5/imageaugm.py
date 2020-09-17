#
#
#
import glob
import cv2
import numpy as np
import random
# from utils.box_utils import matrix_iof
# image augmentator
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug import parameters as iap
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# draw augmentation results on image
def draw_on_image(image, keypnt):


    # drawing bounding box and keypoints for control
    cv2.rectangle(image, (int(keypnt[0]), int(keypnt[1])), (int(keypnt[2]), int(keypnt[3])), (255,0,0),2)
    #
    # cv2.putText(image, '.', (int(keypnt[0][4]), int(keypnt[0][5])), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    # cv2.putText(image, '.', (int(keypnt[0][6]), int(keypnt[0][7])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    # cv2.putText(image, '.', (int(keypnt[0][8]), int(keypnt[0][9])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    # cv2.putText(image, '.', (int(keypnt[0][10]), int(keypnt[0][11])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    return image

# augmentation with imagaug
class ImgAugTransform():
    def __init__(self,  rgb_mean, randomImg, insize):
        sometimes = lambda aug: iaa.Sometimes(0.7, aug)
        self.rand_img_dir = randomImg
        self.rgb_mean = rgb_mean
        self.inp_dim = insize
        #
        self.randomImgList = glob.glob( randomImg + '*.jpg')
        self.supplImgList = glob.glob( '/home/ali/ProjLAB/data/BK4/AAAPistol/*.jpg' )
        self.aug = iaa.Sequential([
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-25, 25), # rotate by -45 to +45 degrees
            shear=(-6, 6), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),

        iaa.OneOf([
            iaa.Fliplr(0.5),

            iaa.GaussianBlur(
                sigma=iap.Uniform(0.0, 1.0)
            ),

            iaa.BlendAlphaSimplexNoise(
                foreground=iaa.BlendAlphaSimplexNoise(
                    foreground=iaa.EdgeDetect(1.0),
                    background=iaa.LinearContrast((0.1, .8)),
                    per_channel=True
                ),
                background=iaa.BlendAlphaFrequencyNoise(
                    exponent=(-.5, -.1),
                    foreground=iaa.Affine(
                        rotate=(-10, 10),
                        translate_px={"x": (-1, 1), "y": (-1, 1)}
                    ),
                    # background=iaa.AddToHueAndSaturation((-4, 4)),
                    # per_channel=True
                ),
                per_channel=True,
                aggregation_method="max",
                sigmoid=False
            ),

        iaa.BlendAlpha(
            factor=(0.2, 0.8),
            foreground=iaa.Sharpen(1.0, lightness=2),
            background=iaa.CoarseDropout(p=0.1, size_px=8)
        ),

        iaa.BlendAlpha(
            factor=(0.2, 0.8),
            foreground=iaa.Affine(rotate=(-5, 5)),
            per_channel=True
        ),
        iaa.MotionBlur(k=15, angle=[-5, 5]),
        iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4),
                                       foreground=iaa.AddToHue((-10, 10))),
        iaa.BlendAlphaElementwise((0, 1.0), iaa.AddToHue(10)),
        iaa.BilateralBlur(
                d=(3, 10), sigma_color=(1, 5), sigma_space=(1, 5)),
        iaa.AdditiveGaussianNoise(scale=0.02 * 255),
        iaa.AddElementwise((-5, 5), per_channel=0.5),
        iaa.AdditiveLaplaceNoise(scale=0.01 * 255),
        iaa.AdditivePoissonNoise(20),
        iaa.Cutout(fill_mode="gaussian", fill_per_channel=True),
        iaa.CoarseDropout(0.02, size_percent=0.1),
        iaa.SaltAndPepper(0.1, per_channel=True),
        iaa.JpegCompression(compression=(70, 99)),
        iaa.ImpulseNoise(0.02),
        iaa.Dropout(p=(0, 0.04)),
        iaa.Sharpen(alpha=0.1),
        ]) # oneof

        ])

    def _resize_subtract_mean(self, image, target ):
        # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        # interp_method = interp_methods[random.randrange(5)]
        # insize = insize[random.randrange(3)]
        # image = cv2.resize(image, (self.insize, self.insize), interpolation=interp_method)
        image = image.astype(np.float32)

        image -= self.rgb_mean
        image /= 255.

        target[:4] = np.array([target[i]/image.shape[(i+1) % 2] for i in range(4) ])

        return image, target # .transpose(2, 0, 1)

    def crop_and_paste(self, src, img, target, robj):
        #
        if not robj:
            objectbbox = src[int(round(target[1])):int(round(target[3])),
                    int(round(target[0])):int(round(target[2]))]
            try:
                rratio_x = np.random.uniform(.8, 1.25)
                rratio_y = np.random.uniform(.8, 1.25)

                rand_x = int(objectbbox.shape[1] * rratio_x)
                rand_y = int(rratio_y * objectbbox.shape[0])
                # selecting a uniform position on the target image
                objectbbox = cv2.resize(objectbbox, (rand_x, rand_y))
                ly, lx = objectbbox.shape[:2]
                #
                plc_x, plc_y = np.random.randint(2, img.shape[1] - lx - 2), np.random.randint(2, img.shape[0] - ly - 2)
                #
                # check = False
            except:
                return None, None
        # random enlargement factor
        else:
            objectbbox = src
            rratio_x = np.random.uniform(.09, .25)
            rratio_y = np.random.uniform(.09, .25)

            rand_x = int(img.shape[1] * rratio_x)
            rand_y = int(rratio_y * img.shape[0])
            # selecting a uniform position on the target image
            objectbbox = cv2.resize(objectbbox, (rand_x, rand_y))
            ly, lx = objectbbox.shape[:2]
            #
            plc_x, plc_y = np.random.randint(2, img.shape[1] - lx - 2), np.random.randint(2, img.shape[0] - ly - 2)

            # target[0][1], target[0][2], target[0][3], target[0][4] = 0,0, src.shape[1], src.shape[0]



        oif = np.abs(np.random.randint(75,100))/100
        img[plc_y: plc_y + ly, plc_x:plc_x+lx] = (1-oif)*img[plc_y: plc_y + ly, plc_x:plc_x+lx] + oif * objectbbox
        # smoothing the crop paste
        img = cv2.GaussianBlur(img, (3,3), 0)

        # modify the target
        # for i in range(4):
        #     target[0][4 + 2*i]     = (target[0][4+2*i] - target[0][0])* rratio_x  + plc_x
        #     target[0][4 + 2*i + 1] = (target[0][4 + 2*i+1] - target[0][1])* rratio_y  + plc_y

        target[0], target[1], target[2], target[3] = plc_x, plc_y, plc_x + lx, plc_y + ly

        return img, target

    def mix_images(self, img_1, img_2):
        #
        landa = np.random.uniform(.65,.85)
        image  = landa * img_1 +  (1-landa) * img_2
        #
        return image


    def get_random_image(self):
        rind = np.random.randint(0, len(self.randomImgList))
        random_img = cv2.imread( self.randomImgList[rind] )
        return random_img

    def letterbox_image(self, img):
        '''resize image with unchanged aspect ratio using padding

        Parameters
        ----------

        img : numpy.ndarray
            Image

        inp_dim: tuple(int)
            shape of the reszied image

        Returns
        -------

        numpy.ndarray:
            Resized image

        '''

        inp_dim = (self.inp_dim, self.inp_dim)
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = inp_dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h))

        canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)

        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

        return canvas

    def yolo_resize(self, img, bboxes):
        w, h = img.shape[1], img.shape[0]
        img = self.letterbox_image(img)

        scale = min(self.inp_dim / h, self.inp_dim / w)
        bboxes[:4] *= (scale)
        new_w = scale * w
        new_h = scale * h
        inp_dim = self.inp_dim

        del_h = (inp_dim - new_h) / 2
        del_w = (inp_dim - new_w) / 2

        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)

        bboxes[:4] += add_matrix[0]

        # img = img.astype(np.uint8)

        return img, bboxes

    def augment_img_lbl(self, img, target):
        # img = np.array(img, dtype='uint8')

        # TODO: feed random images for mixing and cropping
        # img, lndmrk = self.aug.augment_image(img, target)
        # to visualize here!
        # Augment keypoints and images.
        selector = random.randint(0, 10)
        # selector = 0
        if  selector== 0 or selector ==1:
            # 5% select from supplementary gun images
            randImg = cv2.resize(self.get_random_image(), (img.shape[1], img.shape[0]))
            if random.randint(0,20) < 1:
                srcimg = cv2.imread( self.supplImgList[np.random.randint(0,len(self.supplImgList))])
                robj=1
                image_aug, target_crp = self.crop_and_paste(srcimg, randImg, target.copy(), robj)
            else:
                robj = 0
                image_aug, target_crp = self.crop_and_paste(img.copy(), randImg, target.copy(), robj)

            if image_aug is None:
               image_aug = img
               target = target
            else:
                target = target_crp
            # vis_img1 = draw_on_image(image_crp, target_crp)
            # cv2.namedWindow('crop', cv2.WINDOW_NORMAL)
            # cv2.imshow('crop',vis_img1.astype('uint8'))
            # cv2.waitKey(0)

        elif selector==2 :
            randImg = cv2.resize(self.get_random_image(), (img.shape[1], img.shape[0]))
            image_aug = self.mix_images(img.copy(), randImg)

            # vis_img2 = draw_on_image(mixedimg, target.copy())
            # cv2.namedWindow('mix', cv2.WINDOW_NORMAL)
            # cv2.imshow('mix', vis_img2.astype('uint8'))
            # cv2.waitKey(0)

        elif selector in [3,4,5,6]:
            kps = BoundingBoxesOnImage([
                                    BoundingBox(x1=target[0], y1=target[1], x2 = target[2], y2 = target[3] ),
                                    # Keypoint(x=target[0][2], y=target[0][3])
                                    # Keypoint(x=target[0][8], y=target[0][9]),
                                    # Keypoint(x=target[0][10], y=target[0][11])
                                ], shape=img.shape)
            # creting bounding box based on augmented landmarks
            image_aug, kps_aug = self.aug(image=img.copy(), bounding_boxes=kps)
            tries = 4
            cntr = 0
            while ( kps_aug[0].x1<0 or kps_aug[0].y1<0 or kps_aug[0].x2<0 or kps_aug[0].y2<0 or \
                    kps_aug[0].x1> img.shape[1] or kps_aug[0].y1> img.shape[0] or kps_aug[0].x2>img.shape[1] or kps_aug[0].y2>img.shape[0] ):
                # print('found negative keypoints!')
                image_aug, kps_aug = self.aug(image=img.copy(), bounding_boxes=kps)
                cntr+=1
                if cntr>tries:
                    break
            #
            if cntr<4:
                target[0], target[1], target[2], target[3] = kps_aug[0].x1, kps_aug[0].y1, kps_aug[0].x2, kps_aug[0].y2

            # vis_img3 = draw_on_image(image_aug, target.copy())
            # cv2.namedWindow('last', cv2.WINDOW_NORMAL)
            # cv2.imshow('last', vis_img3.astype('uint8'))
            # cv2.waitKey(0)

            # xmin, ymin, xmax, ymax = img.shape[1], img.shape[0], 0, 0
            # TODO: dividing landmarks and bounding boxes to the width and height
            # for i in range(4):
            #     target[0][4 + 2 * i], target[0][4 + 2 * i + 1] = kps_aug[i].x , kps_aug[i].y
            #     # print('keypoint x: ', kps_aug[i].x, 'keypoint y: ', kps_aug[i].y)
            #     if xmin > kps_aug[i].x:
            #         xmin = kps_aug[i].x
            #     if xmax < kps_aug[i].x:
            #         xmax = kps_aug[i].x
            #     if ymin > kps_aug[i].y:
            #         ymin = kps_aug[i].y
            #     if ymax < kps_aug[i].y:
            #         ymax = kps_aug[i].y

            # target[0][0], target[0][1] = (xmin - 5), (ymin - 5)
            # target[0][2], target[0][3] = (xmax + 5), (ymax + 5)


        else:
            image_aug = img


        # draw_on_image(image_aug, target)


        image_aug, target = self.yolo_resize( image_aug, target )

        image_aug, target = self._resize_subtract_mean( image_aug, target )

        # vis_img2 = draw_on_image(image_aug.astype('uint8'), target.copy())
        # cv2.namedWindow('fin', cv2.WINDOW_FREERATIO)
        # cv2.imshow('fin', vis_img2.astype('uint8'))
        # cv2.waitKey(0)

        return image_aug, target



if __name__ == '__main__':
    #
    rgb_mean = (104, 117, 123)  # bgr order
    img_dim = 640
    # daug = preproc( img_dim, rgb_mean )
    #
    transforms_imgaug = ImgAugTransform()
    #






