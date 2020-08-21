import os
import numpy as np

import MapExtrackt

class ExtractFeatureMaps:

    def __init__(self, model ,display_every_n = 30):

        self.model = model
        self.display_every_n = display_every_n

        #self._extraction()


    def extraction(self, image_name):
        self.image_name = image_name
        fe = MapExtrackt.FeatureExtractor(self.model)
        fe.set_image(self.image_name)

        conv2d_layers_indexes = [i for i,x in enumerate(fe.layer_names) if x == "Conv2d"]

        desired_indexes = conv2d_layers_indexes[::self.display_every_n]

        all_images = []

        for index in desired_indexes:

            all_images.append(np.array(fe[index]))


        # print("from within map extract, type all images", type(all_images_stacked))
        return np.asarray(all_images)




