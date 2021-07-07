"""Texture Networks: Feed-forward Synthesis of Textures and Stylized Images

Related papers
- https://arxiv.org/abs/1603.03417

trains compact feed-forward convolutional networks to generate multiple samples of the same texture of arbitrary size

using circular convolution to remove boundary effects, which is appropriate for textures

upsampling layers use simple nearest-neighbour interpolation, (we also experimented strided full-convolution,
but the results were not satisfying).

we found that training benefited significantly from inserting batch normalization layers
right after each convolutional layer and, most importantly, right before the concatenation layers,
since this balances gradients travelling along different branches of the network.
"""