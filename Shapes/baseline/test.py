# -*- coding: utf-8 -*-
"""
Generates similarities, images and MDS vectors that can be used for testing our correlation scripts.

Created on Wed Dec 19 10:43:40 2018

@author: lbechberger
"""

import numpy as np
import pickle
from PIL import Image

# ground truth: dissim(a,b) = 1, dissim(a,c) = 2, dissim(b,c) = 3
item_ids = ['A','B','C']
dissimilarities = np.array([[0,1,2],[1,0,3],[2,3,0]])

result = {'items': item_ids, 'dissimilarities': dissimilarities}
with open('test/test_dissimilarities.pickle', 'wb') as f:
    pickle.dump(result, f)

# now create some good MDS vectors
with open('test/vectors-good/1D-vectors.csv', 'w') as f:
    f.write('A,0\n');
    f.write('B,-0.1\n');
    f.write('C,0.3\n');

with open('test/vectors-good/2D-vectors.csv', 'w') as f:
    f.write('A,0,0\n');
    f.write('B,10,10\n');
    f.write('C,-20,-20\n');


# now create some bad MDS vectors
with open('test/vectors-bad/1D-vectors.csv', 'w') as f:
    f.write('A,0\n');
    f.write('B,1\n');
    f.write('C,2\n');

with open('test/vectors-bad/2D-vectors.csv', 'w') as f:
    f.write('A,0,0\n');
    f.write('B,0.1,0.1\n');
    f.write('C,0.1,0.11\n');


# now create some good images
rand_img = np.random.uniform(low=100, high=150, size=[100, 100])
img_a = rand_img
img_b = rand_img + 50 * np.ones(shape=(100,100))
img_c = rand_img - 100 * np.ones(shape=(100,100))

pil_a = Image.fromarray(img_a)
pil_a = pil_a.convert("L")
pil_a.save('test/images-good/A.jpg')

pil_b = Image.fromarray(img_b)
pil_b = pil_b.convert("L")
pil_b.save('test/images-good/B.jpg')

pil_c = Image.fromarray(img_c)
pil_c = pil_c.convert("L")
pil_c.save('test/images-good/C.jpg')

# now create some bad images
rand_img = np.random.uniform(low=100, high=150, size=[100, 100])
img_a = rand_img
img_b = rand_img + np.random.uniform(low= -100, high=100, size=(100,100))
img_c = rand_img + np.random.uniform(low= -100, high=100, size=(100,100))

pil_a = Image.fromarray(img_a)
pil_a = pil_a.convert("L")
pil_a.save('test/images-bad/A.jpg')

pil_b = Image.fromarray(img_b)
pil_b = pil_b.convert("L")
pil_b.save('test/images-bad/B.jpg')

pil_c = Image.fromarray(img_c)
pil_c = pil_c.convert("L")
pil_c.save('test/images-bad/C.jpg')
