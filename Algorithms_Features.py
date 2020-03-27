import cv2
import argparse

Feature_descriptor_algorithms = {'SIFT': cv2.xfeatures2d.SIFT_create(),
                                 'SURF': cv2.xfeatures2d.SURF_create(),
                                 'ORB': cv2.ORB_create(),
                                 'BRISK': cv2.BRISK_create(),
                                 'BRIEF': cv2.xfeatures2d.BriefDescriptorExtractor_create(),
                                 'AKAZE': cv2.AKAZE_create()}


def display_img(img, name="image"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_feature(image, algorithms, algo):

    if algo == 'BRIEF':
        # Initiate FAST detector
        # find the key points with STAR
        kp = cv2.xfeatures2d.StarDetector_create().detect(image, None)
        # compute the descriptors with BRIEF
        kp, desc = algorithms[algo].compute(image, kp)
    else:
        kp, desc = algorithms[algo].detectAndCompute(image, None)

    cv2.drawKeypoints(image, kp, image)
    display_img(image)

    return image, desc


if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-path", "--path_image", type=str,
                    default='images/vibot.jpg',
                    help="path the image .txt format")
    ap.add_argument("-name_algo", "--algorithm_name", type=str,
                    default="SIFT",
                    help="Chose Name of algorithm [SIFT, SURF, ORB, BRISK, BRIEF, AKAZE] string format ")
    args = vars(ap.parse_args())

    img = cv2.imread(args["path_image"])

    get_feature(img, Feature_descriptor_algorithms, args["algorithm_name"])
