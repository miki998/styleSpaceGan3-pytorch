### Quick Descriptions

When discovering the project, we recommend to go through notebooks in the below listed order:

```
generateInfos.ipynb         : generate images (and their ys codes) + generate gradient maps of those images + generate semantic masks of those images
rankYs.ipynb                : from semantic masks and gradient maps output iou and count scores for ranking channels in a layer
manipStyle.ipynb            : through observation of edit results we find best functionning channels within top3 and across layers + example of edits
invert.ipynb                : inversion of real images and editing on real images
iouscore.ipynb              : comparing models' gradient maps' iou scores 
editingScore.ipynb          : attempt on quantifying quality of edits through spatial change (union over intersection) or discriminant
train_lelsd_stylegan2.ipynb : obtaining directions for editing with a LELSD method instead of StyleSpace
manipLelsd.ipynb            : do editing with the LELSD obtained directions and qualitatively see performance
```
