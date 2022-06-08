# ECE228_Project
ECE228_Project Analysis of Neural Style Transfer

"NST.ipynb" contains basic implementation (not all, some experiments are taken in different .ipynb such as different pretrained model).
It has three sections. First, Neural style transfer implementation. Second, Feature map visualization. Third, Experiments with different hyperparameters.

To run "NST.ipynb" the path should be:
```
---- Main directory ----

-NST.ipynb

-feature_map_extraction.py

-mona.jpg <- used as example for style image

-pp1.jpg <- used as example for content image
```
Run ```NST.ipynb``` from top to bottom should reproduce same results as showed in report.

# Part one (NST implementation)
## pretrained model
We need pretrained model to do feature extraction, and we used pretrained VGG-19 provided by torchvision.

```vgg = models.vgg19(pretrained=True)```

## select input image (content and style)
Load content and style image. In this example, we use pp1.jpg as content image and 'mona.jpg' as style image.

```content = load_image('pp1.jpg').to(device)```

```style = load_image('mona.jpg', shape=content.shape[-2:]).to(device)```
![GitHub Logo](/image/NST_input.png)

## layer selection and target selection
Select specific layers from VGG-19 to extract features for both content and style image.

```content_features = get_features(content, vgg)```

```style_features = get_features(style, vgg)```

We can choose content image as target or choose random noised input as target.

```target = content.clone().requires_grad_(True).to(device)``` <- in this part, we choose content as initial target.

```target = torch.rand(1,3,400,533, requires_grad=True, device="cuda")```

## NST training
Perform gradient descend on input (target) to do Neural style transfer.

![GitHub Logo](/image/NST_result.png)

# Part two (Feature map visualization)
To understand what kind of feature a layer represents, we visualize the feature map in that layer.
There are plenty of feature map in one layer (decided by number of channel). For simplicity, I only display 4 maps.

```layer_selection = ['2','5','8','11','15']```, select different layers to see difference.

```index = 0``` <- visualize layer_selection[0] = ```'2'``` in VGG-19
![GitHub Logo](/image/moan_fmap_0.png)

```index = 4``` <- visualize layer_selection[4] = ```'15'``` in VGG-19
![GitHub Logo](/image/moan_fmap_4.png)

# Part three (Experiments with different hyperparameters)
In this experiment, we set up four different hyperparameters.
![GitHub Logo](/image/Parameter_set.JPG)
## Log(loss) curve on four different sets.
![GitHub Logo](/image/loss_curve.png)

## Sequential output on four different sets.
![GitHub Logo](/image/result_small_image.png)

# Other jupyter notebook (other experiments)
We choose two different pretrained model to see the influence of different structures.

To run these .ipynb, we need to upgrade pytorch version by following command.

```
!pip install --upgrade torchvision==0.12
```

## transfer-EfficientNet.ipynb
In this part, we use pretrained EfficientNet provided by torchvision.

Procedure is similar to ```NST.ipynb```, therefore, we only show the results here.
![GitHub Logo](/image/result_efficientnet.png)

## transfer-Convnext.ipynb
In this part, we use pretrained ConvNeXt provided by torchvision.

Procedure is similar to ```NST.ipynb```, therefore, we only show the results here.
![GitHub Logo](/image/result_convnext.png)
