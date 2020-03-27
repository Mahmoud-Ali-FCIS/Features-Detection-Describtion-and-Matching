import cv2
import matplotlib.pyplot as plt
import argparse

Feature_descriptor_algorithms = {'SIFT': cv2.xfeatures2d.SIFT_create(),
                                 'SURF': cv2.xfeatures2d.SURF_create(),
                                 'ORB': cv2.ORB_create(),
                                 'BRISK': cv2.BRISK_create(),
                                 'BRIEF': cv2.xfeatures2d.BriefDescriptorExtractor_create(),
                                 'AKAZE': cv2.AKAZE_create()}


def feature_matching(algorithms, algo, img1, img2):

    if algo == 'BRIEF':
        # Initiate FAST detector
        # find the key points with STAR
        kp1 = cv2.xfeatures2d.StarDetector_create().detect(img1, None)
        # compute the descriptors with BRIEF
        kp1, des1 = algorithms[algo].compute(img1, kp1)
        kp2 = cv2.xfeatures2d.StarDetector_create().detect(img2, None)
        # compute the descriptors with BRIEF
        kp2, des2 = algorithms[algo].compute(img2, kp2)
    else:
        kp1, des1 = algorithms[algo].detectAndCompute(img1, None)
        kp2, des2 = algorithms[algo].detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    threshold = 110
    good_match = []

    for match in matches:
        if match.distance < threshold:
            good_match.append(match)

    img3 = cv2.drawMatches(img1, kp1,
                           img2, kp2,
                           good_match,
                           flags=2, outImg=None)

    plt.imshow(img3)
    plt.show()


if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-path1", "--path_image1", type=str,
                    default='images/vibot.jpg',
                    help="path the queryImage .txt format")
    ap.add_argument("-path2", "--path_image2", type=str,
                    default='images/vibot_in_scene.png',
                    help="path the trainImage .txt format")
    ap.add_argument("-name_algo", "--algorithm_name", type=str,
                    default="SIFT",
                    help="Chose Name of algorithm [SIFT, SURF, ORB, BRISK, BRIEF, AKAZE] string format ")
    args = vars(ap.parse_args())

    img1 = cv2.imread(args["path_image1"])  # queryImage
    img2 = cv2.imread(args["path_image2"])  # trainImage

    feature_matching(Feature_descriptor_algorithms, args["algorithm_name"], img1, img2)
