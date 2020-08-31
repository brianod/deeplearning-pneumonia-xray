**Preface**

I am using the chest x-ray dataset here: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

I have placed it in the root of my Google Drive which is where this notebook on Colaboratory will access it from.

On Google Colaboratory, sure you select Runtime -> Change Runtime Type -> GPU before running the notebook.

**Setup**


```
#hide
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```

    [K     |████████████████████████████████| 61kB 4.3MB/s 
    [K     |████████████████████████████████| 51kB 5.3MB/s 
    [K     |████████████████████████████████| 1.0MB 8.4MB/s 
    [K     |████████████████████████████████| 358kB 12.7MB/s 
    [K     |████████████████████████████████| 40kB 6.5MB/s 
    [K     |████████████████████████████████| 92kB 10.5MB/s 
    [K     |████████████████████████████████| 61kB 8.3MB/s 
    [K     |████████████████████████████████| 51kB 8.0MB/s 
    [K     |████████████████████████████████| 2.7MB 18.9MB/s 
    [?25hGo to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive



```
#hide
from fastbook import *
from fastai.vision.widgets import *
```

**Set the path of the chest_xray folder in Google Drive**


```
xray_path = Path.joinpath(gdrive, 'chest_xray')
xray_path.is_dir()
```




    True



**Verify the chest_xray image files are valid images**


```
fns = get_image_files(xray_path)
fns
```




    (#5856) [Path('/content/gdrive/My Drive/chest_xray/val/PNEUMONIA/person1951_bacteria_4882.jpeg'),Path('/content/gdrive/My Drive/chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg'),Path('/content/gdrive/My Drive/chest_xray/val/PNEUMONIA/person1949_bacteria_4880.jpeg'),Path('/content/gdrive/My Drive/chest_xray/val/PNEUMONIA/person1947_bacteria_4876.jpeg'),Path('/content/gdrive/My Drive/chest_xray/val/PNEUMONIA/person1952_bacteria_4883.jpeg'),Path('/content/gdrive/My Drive/chest_xray/val/PNEUMONIA/person1950_bacteria_4881.jpeg'),Path('/content/gdrive/My Drive/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg'),Path('/content/gdrive/My Drive/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg'),Path('/content/gdrive/My Drive/chest_xray/val/NORMAL/NORMAL2-IM-1437-0001.jpeg'),Path('/content/gdrive/My Drive/chest_xray/val/NORMAL/NORMAL2-IM-1442-0001.jpeg')...]




```
failed = verify_images(fns)
failed
```








    (#0) []



**Specify folders containing training, validation, and test sets**


```
custom_splitter = GrandparentSplitter(train_name=('train', 'val'), valid_name='test')
```

**Create a DataBlock and DataLoaders**

A DataBlock is the transformed data, and a DataLoader in PyTorch is an abstraction that makes it easier to iterate over data using parameters such as batch size.


```
xrays = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=custom_splitter,
    get_y=parent_label,
    item_tfms=Resize(256))
```


```
dls = xrays.dataloaders(xray_path)
```

**Show 4 chest x-rays from the validation set.**


```
dls.valid.show_batch(max_n=4, nrows=1)
```


![png](Pneumonia_files/Pneumonia_15_0.png)


**Randomly crop to different areas of the xray**

More data augmentation. The following are the same xray shown with 4 different crops.


```
xrays = xrays.new(item_tfms=RandomResizedCrop(256, min_scale=0.7))
dls = xrays.dataloaders(xray_path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```


![png](Pneumonia_files/Pneumonia_17_0.png)


**Use resnet50 pre-trained model and find the optimal learning rate**

The optimal learning rate is an order of magnitude less than the lowest point of the loss.


```
learn = cnn_learner(dls, resnet50, metrics=error_rate)
learn.lr_find()
```

    Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth



    HBox(children=(FloatProgress(value=0.0, max=102502400.0), HTML(value='')))


    









    SuggestedLRs(lr_min=0.00831763744354248, lr_steep=0.0006918309954926372)




![png](Pneumonia_files/Pneumonia_19_5.png)


**Fine tune the pretrained model by training the last layers for 4 epochs**

Transfer learning. 4 epochs was an arbitrary choice. 

`fine_tune` freezes the well-trained first layers, trains for `freeze_epochs=1` epochs, then unfreezes and trains all layers for 4 (argument supplied) epochs.


```
learn.fine_tune(4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.454557</td>
      <td>0.718677</td>
      <td>0.198718</td>
      <td>01:43</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.188958</td>
      <td>1.042401</td>
      <td>0.245192</td>
      <td>01:51</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.119776</td>
      <td>0.607215</td>
      <td>0.217949</td>
      <td>01:50</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.078995</td>
      <td>0.627241</td>
      <td>0.187500</td>
      <td>01:49</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.053839</td>
      <td>0.376141</td>
      <td>0.121795</td>
      <td>01:50</td>
    </tr>
  </tbody>
</table>


Unfreeze all layers and train for 12 epochs using the learning rate we found with `lr_find`


```
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-4,1e-2))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.114177</td>
      <td>0.451784</td>
      <td>0.123397</td>
      <td>01:50</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.142159</td>
      <td>1.167055</td>
      <td>0.203526</td>
      <td>01:49</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.205139</td>
      <td>0.360007</td>
      <td>0.120192</td>
      <td>01:50</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.158097</td>
      <td>0.312015</td>
      <td>0.070513</td>
      <td>01:50</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.115267</td>
      <td>0.336613</td>
      <td>0.105769</td>
      <td>01:50</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.112579</td>
      <td>1.180396</td>
      <td>0.224359</td>
      <td>01:49</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.097856</td>
      <td>0.354895</td>
      <td>0.105769</td>
      <td>01:48</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.064272</td>
      <td>0.735087</td>
      <td>0.166667</td>
      <td>01:48</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.050575</td>
      <td>0.693501</td>
      <td>0.149038</td>
      <td>01:47</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.033466</td>
      <td>0.634683</td>
      <td>0.129808</td>
      <td>01:48</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.025155</td>
      <td>0.556284</td>
      <td>0.120192</td>
      <td>01:47</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.023254</td>
      <td>0.536361</td>
      <td>0.115385</td>
      <td>01:48</td>
    </tr>
  </tbody>
</table>


**~88% accuracy!**

Plot the confusion matrix. It seems the model's errors are almost always false positives rather than false negatives. That is, it predicts someone has Pneumonia when they don't, but rarely predicts that someone who does have Pneuomonia doesn't have Pneumonia. False positives are better than false negatives in this case, at least virtually no one with pneumonia is denied treatment.


```
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```






![png](Pneumonia_files/Pneumonia_25_1.png)


Let's plot images the model got wrong.


```
interp.plot_top_losses(10, nrows=2)
```


![png](Pneumonia_files/Pneumonia_27_0.png)


P.S. This readme was created using nbconvert --to markdown [file_name].ipynb
