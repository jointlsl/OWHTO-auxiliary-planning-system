import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from   imgaug.augmentables.segmaps import SegmentationMapsOnImage
from   imgaug.augmentables import Keypoint, KeypointsOnImage


def transformer(img_arr, kpts_arr):
    
    img_arr     = np.transpose(img_arr, (2,1,0))

    kpts_list = [Keypoint(x=kpt[0], y=kpt[1]) for kpt in kpts_arr]
    kps = KeypointsOnImage(kpts_list, shape=img_arr.shape)

    # seq = iaa.Sequential([
    #                     # Small gaussian blur with random sigma between 0 and 0.5.
    #                     # But we only blur about 50% of all images.
    #                     iaa.Sometimes(
    #                         0.5,
    #                         iaa.GaussianBlur(sigma=(0, 0.5))
    #                     ),
    #                     # Strengthen or weaken the contrast in each image.
    #                     iaa.LinearContrast((0.75, 1.5)),
    #                     # Add gaussian noise.
    #                     # For 50% of all images, we sample the noise once per pixel.
    #                     # For the other 50% of all images, we sample the noise per pixel AND
    #                     # channel. This can change the color (not only brightness) of the
    #                     # pixels.
    #                     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    #                     iaa.Multiply((0.8, 1.2)), # change brightness, doesn't affect keypoints
    #                     # Apply affine transformations to each image.
    #                     # Scale/zoom them, translate/move them, rotate them and shear them.
    #                     iaa.Affine(
    #                         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #                         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #                         rotate=(-10, 10),
    #                         shear=(-8, 8)
    #                     )
    #                     ], random_order=True)

    seq = iaa.SomeOf((0, 3), [
                        # Small gaussian blur with random sigma between 0 and 0.5.
                        # But we only blur about 50% of all images.
                        iaa.Sometimes(
                            0.5,
                            iaa.GaussianBlur(sigma=(0, 0.5))
                        ),
                        # Strengthen or weaken the contrast in each image.
                        iaa.LinearContrast((0.75, 1.5)),
                        # Add gaussian noise.
                        # For 50% of all images, we sample the noise once per pixel.
                        # For the other 50% of all images, we sample the noise per pixel AND
                        # channel. This can change the color (not only brightness) of the
                        # pixels.
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                        iaa.Multiply((0.8, 1.2)), # change brightness, doesn't affect keypoints
                        # Apply affine transformations to each image.
                        # Scale/zoom them, translate/move them, rotate them and shear them.
                        iaa.Affine(
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                            rotate=(-10, 10),
                            shear=(-8, 8)
                        )
                        ])

    img_aug, kps_aug = seq(image=img_arr, keypoints=kps)

    img_aug_arr = np.transpose(img_aug, (2,1,0))
    kps_aug_arr = np.array([[i.x, i.y] for i in kps_aug.keypoints])

    return img_aug_arr, kps_aug_arr


    # print(img_aug.shape)
    # print(kps_aug.keypoints)
    # # print coordinates before/after augmentation (see below)
    # # use after.x_int and after.y_int to get rounded integer coordinates
    # for i in range(len(kps.keypoints)):
    #     before = kps.keypoints[i]
    #     after = kps_aug.keypoints[i]
    #     print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
    #         i, before.x, before.y, after.x, after.y)
    #     )

    # image with keypoints before/after augmentation (shown below)
    # image_before = kps.draw_on_image(img_arr, size=7)
    # image_after = kps_aug.draw_on_image(img_aug, size=7)
    # print(image_before.shape, image_after.shape)

    # import matplotlib.pyplot as plt

    # # 假设你的代码已经创建了 `image_before` 和 `image_after`

    # # 显示图像
    # plt.figure(figsize=(10, 5))  # 设置图像大小

    # # 显示原始图像和关键点
    # plt.subplot(1, 2, 1)  # 1行2列的第1个
    # plt.title('Before Augmentation')
    # plt.imshow(image_before)

    # # 显示增强后的图像和关键点
    # plt.subplot(1, 2, 2)  # 1行2列的第2个
    # plt.title('After Augmentation')
    # plt.imshow(image_after)

    # plt.show()  # 显示图像

    # import cv2
    # cv2.imshow('123', image_before)
    # cv2.imshow('345', image_after)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    import sys
    sys.exit(0)

    # sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # seq = iaa.Sequential(
    #     [
    #     ##
    #     ## Apply the following augmenters to most images.
    #     ##
    #     # iaa.Fliplr(0.5), # horizontally flip 50% of all images
    #     # iaa.Flipud(0.2), # vertically flip 20% of all images

    #     ### crop some of the images by 0-10% of their height/width
    #     # sometimes(iaa.Crop(percent=(0, 0.1))),

    #     ## Apply affine transformations to some of the images
    #     ## - scale to 80-120% of image height/width (each axis independently)
    #     ## - translate by -20 to +20 relative to height/width (per axis)
    #     ## - rotate by -45 to +45 degrees
    #     ## - shear by -16 to +16 degrees
    #     ## - order: use nearest neighbour or bilinear interpolation (fast)
    #     ## - mode: use any available mode to fill newly created pixels
    #     ##         see API or scikit-image for which modes are available
    #     ## - cval: if the mode is constant, then use a random brightness
    #     ##        for the newly created pixels (e.g. sometimes black,
    #     ##         sometimes white)
    #     # sometimes(iaa.Affine(
    #     #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     #     rotate=(-45, 45),
    #     #     shear=(-16, 16),
    #     #     order=[0, 1],
    #     #     cval=(0, 255),
    #     #     mode=ia.ALL
    #     # )),

    #     ##
    #     ## Execute 0 to 5 of the following (less important) augmenters per
    #     ## image. Don't execute all of them, as that would often be way too
    #     ## strong.
    #     ##
    #     iaa.SomeOf((0, 5),
    #         [
    #             # Convert some images into their superpixel representation,
    #             # sample between 20 and 200 superpixels per image, but do
    #             # not replace all superpixels with their average, only
    #             # some of them (p_replace).
    #             # sometimes(
    #             #     iaa.Superpixels(
    #             #         p_replace=(0, 1.0),
    #             #         n_segments=(20, 200)
    #             #     )
    #             # ),

    #             # Blur each image with varying strength using
    #             # gaussian blur (sigma between 0 and 3.0),
    #             # average/uniform blur (kernel size between 2x2 and 7x7)
    #             # median blur (kernel size between 3x3 and 11x11).
    #             iaa.OneOf([
    #                 iaa.GaussianBlur((0, 3.0)),
    #                 iaa.AverageBlur(k=(2, 7)),
    #                 iaa.MedianBlur(k=(3, 11)),
    #             ]),

    #             # Sharpen each image, overlay the result with the original
    #             # image using an alpha between 0 (no sharpening) and 1
    #             # (full sharpening effect).
    #             iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

    #             # # Same as sharpen, but for an embossing effect.
    #             # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

    #             # # Search in some images either for all edges or for
    #             # # directed edges. These edges are then marked in a black
    #             # # and white image and overlayed with the original image
    #             # # using an alpha of 0 to 0.7.
    #             # sometimes(iaa.OneOf([
    #             #     iaa.EdgeDetect(alpha=(0, 0.7)),
    #             #     iaa.DirectedEdgeDetect(
    #             #         alpha=(0, 0.7), direction=(0.0, 1.0)
    #             #     ),
    #             # ])),

    #             # Add gaussian noise to some images.
    #             # In 50% of these cases, the noise is randomly sampled per
    #             # channel and pixel.
    #             # In the other 50% of all cases it is sampled once per
    #             # pixel (i.e. brightness change).
    #             iaa.AdditiveGaussianNoise(
    #                 loc=0, scale=(0.0, 0.05*255), per_channel=0.5
    #             ),

    #             # Either drop randomly 1 to 10% of all pixels (i.e. set
    #             # them to black) or drop them on an image with 2-5% percent
    #             # of the original size, leading to large dropped
    #             # rectangles.
    #             iaa.OneOf([
    #                 iaa.Dropout((0.01, 0.1), per_channel=0.5),
    #                 iaa.CoarseDropout(
    #                     (0.03, 0.15), size_percent=(0.02, 0.05),
    #                     per_channel=0.2
    #                 ),
    #             ]),

    #             # # Invert each image's channel with 5% probability.
    #             # # This sets each pixel value v to 255-v.
    #             # iaa.Invert(0.05, per_channel=True), # invert color channels

    #             # # Add a value of -10 to 10 to each pixel.
    #             iaa.Add((-10, 10), per_channel=0.5),

    #             # # Change brightness of images (50-150% of original value).
    #             iaa.Multiply((0.5, 1.5), per_channel=0.5),

    #             # # Improve or worsen the contrast of images.
    #             # iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

    #             # Convert each image to grayscale and then overlay the
    #             # result with the original with random alpha. I.e. remove
    #             # colors with varying strengths.
    #             iaa.Grayscale(alpha=(0.0, 1.0)),

    #             # # In some images move pixels locally around (with random
    #             # # strengths).
    #             # sometimes(
    #             #     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
    #             # ),

    #             # # In some images distort local areas with varying strength.
    #             # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
    #         ],
    #         ##do all of the above augmentations in random order
    #         random_order=True
    #     )

    # ])

    # img_arr     = np.transpose(img_arr,    (2,1,0))

    # # img_aug_arr, heatmap_aug_arr = seq(image=img_arr, heatmaps=heatmap_arr)


    # img_aug_arr     = np.transpose(img_aug_arr,       (2,1,0))

    # return img_aug_arr
##ondef

def transformer_v02(img_arr, heatmap_arr):

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
        ##
        ## Apply the following augmenters to most images.
        ##
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images

        ### crop some of the images by 0-10% of their height/width
        sometimes(iaa.Crop(percent=(0, 0.1))),

        ## Apply affine transformations to some of the images
        ## - scale to 80-120% of image height/width (each axis independently)
        ## - translate by -20 to +20 relative to height/width (per axis)
        ## - rotate by -45 to +45 degrees
        ## - shear by -16 to +16 degrees
        ## - order: use nearest neighbour or bilinear interpolation (fast)
        ## - mode: use any available mode to fill newly created pixels
        ##         see API or scikit-image for which modes are available
        ## - cval: if the mode is constant, then use a random brightness
        ##        for the newly created pixels (e.g. sometimes black,
        ##         sometimes white)
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),

        ##
        ## Execute 0 to 5 of the following (less important) augmenters per
        ## image. Don't execute all of them, as that would often be way too
        ## strong.
        ##
        iaa.SomeOf((0, 5),
            [
                # Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image, but do
                # not replace all superpixels with their average, only
                # some of them (p_replace).
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # # Same as sharpen, but for an embossing effect.
                # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # # Search in some images either for all edges or for
                # # directed edges. These edges are then marked in a black
                # # and white image and overlayed with the original image
                # # using an alpha of 0 to 0.7.
                # sometimes(iaa.OneOf([
                #     iaa.EdgeDetect(alpha=(0, 0.7)),
                #     iaa.DirectedEdgeDetect(
                #         alpha=(0, 0.7), direction=(0.0, 1.0)
                #     ),
                # ])),

                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

                # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                # # Invert each image's channel with 5% probability.
                # # This sets each pixel value v to 255-v.
                # iaa.Invert(0.05, per_channel=True), # invert color channels

                # # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),

                # # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # # Improve or worsen the contrast of images.
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # # In some images move pixels locally around (with random
                # # strengths).
                # sometimes(
                #     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                # ),

                # # In some images distort local areas with varying strength.
                # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            ##do all of the above augmentations in random order
            random_order=True
        )

    ])

    img_arr     = np.transpose(img_arr,    (2,1,0))
    heatmap_arr = np.transpose(heatmap_arr,(2,1,0))
    heatmap_arr = np.expand_dims(heatmap_arr, axis=0)

    img_aug_arr, heatmap_aug_arr = seq(image=img_arr, heatmaps=heatmap_arr)


    img_aug_arr     = np.transpose(img_aug_arr,       (2,1,0))
    heatmap_aug_arr = np.transpose(heatmap_aug_arr[0],(2,1,0))

    return img_aug_arr, heatmap_aug_arr
##ondef


def transformer4seg2d(img_arr, seg_arr):

    sometimes = lambda aug: iaa.Sometimes(0.2, aug)

    seq = iaa.Sequential(
        [

        ## Apply affine transformations to some of the images
        ## - scale to 80-120% of image height/width (each axis independently)
        ## - translate by -20 to +20 relative to height/width (per axis)
        ## - rotate by -45 to +45 degrees
        ## - shear by -16 to +16 degrees
        ## - order: use nearest neighbour or bilinear interpolation (fast)
        ## - mode: use any available mode to fill newly created pixels
        ##         see API or scikit-image for which modes are available
        ## - cval: if the mode is constant, then use a random brightness
        ##         for the newly created pixels (e.g. sometimes black,
        ##         sometimes white)
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            rotate=(-45, 45),
            shear=(-16, 16),
        )),

        ##
        ## Apply the following augmenters to most images.
        # iaa.Fliplr(1),   # horizontally flip 50% of all images
        # iaa.Flipud(0.2), # vertically flip 20% of all images

        ## crop some of the images by 0-10% of their height/width
        sometimes(iaa.Crop(percent=(0, 0.15))),

        ## In some images move pixels locally around (with random
        ## strengths).
        # sometimes(iaa.ElasticTransformation(alpha=(0, 70), sigma=(4, 6))),


        ## Add gaussian noise to some images.
        ## In 50% of these cases, the noise is randomly sampled per
        ## channel and pixel.
        ## In the other 50% of all cases it is sampled once per
        ## pixel (i.e. brightness change).
        ## sometimes(iaa.AdditiveGaussianNoise(
        ##         scale=(0.0, 0.01)
        ## )),

        ## Blur each image with varying strength using
        ## gaussian blur (sigma between 0 and 3.0),
        ## average/uniform blur (kernel size between 2x2 and 7x7)
        ## median blur (kernel size between 3x3 and 11x11).
        sometimes(iaa.OneOf([
            iaa.GaussianBlur((0, 1.0)),
            iaa.AverageBlur(k=(2, 7)),
        ])),

        ## Either drop randomly 1 to 10% of all pixels (i.e. set
        ## them to black) or drop them on an image with 2-5% percent
        ## of the original size, leading to large dropped
        ## rectangles.
        sometimes(iaa.OneOf([
            iaa.Dropout((0.01, 0.2)),
            iaa.CoarseDropout(
            (0.03, 0.15), size_percent=(0.01, 0.03),
            ),
        ])),

        ## Sharpen each image, overlay the result with the original
        ## image using an alpha between 0 (no sharpening) and 1
        ## (full sharpening effect).
        ## sometimes(iaa.Sharpen(alpha=(0, 0.2), lightness=(0.1, 0.5))),

        ## Improve or worsen the contrast of images.
        sometimes(iaa.LinearContrast((0.6, 1.4))),

    ])


    img_arr = np.transpose(img_arr, (1,2,0))

    segmap = SegmentationMapsOnImage(seg_arr, shape=img_arr.shape)
    image_aug_arr, segmap_aug = seq(image=img_arr, segmentation_maps=segmap)
    segmap_aug_arr = segmap_aug.get_arr()

    image_aug_arr = np.transpose(image_aug_arr, (2,0,1))

    return image_aug_arr, segmap_aug_arr
##ondef