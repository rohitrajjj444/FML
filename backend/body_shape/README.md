# body_shape — Body Shape Prediction module

Owner ke `main.py` se call hota hai. Photo, UI aur baaki models owner ke paas hain. Ye module **sirf ek cheez** deta hai: **body shape (ek string)**.

## Install
```bash
pip install -r requirements.txt
```
`body_shape/` folder ko project me copy karo (`main.py` ke bagal me). `body_shape/weights/body_shape_front.pt` maujood honi chahiye.
Pehli baar chalne par DeepLabV3 (person segmentation, ~160 MB) internet se download hota hai.

## Use (main.py me)
```python
from body_shape import predict_body_shape, get_model

get_model()

body_shape = predict_body_shape(body_photo_bytes, gender)
```
`gender` default `"male"` hai. Women select hone par `"female"` (ya `"women"`) bhejo.
Async framework me `await upload.read()` karke **bytes** pass karo.

## Input
| Field | Type | Note |
|---|---|---|
| image | bytes / PIL / path | Saamne se poori body, sir se paer tak, seedha khada, haath thode alag, saaf background |
| gender | "male" (default) / "female" | |

Height aur weight ki zaroorat nahi hai.

## Output
Sirf ek string, in 5 me se ek:
`"apple"`, `"hourglass"`, `"inverted_triangle"`, `"pear"`, `"rectangle"`

`None` aaye to insaan detect nahi hua ya photo me dikkat hai (sir/paer kate, insaan bahut chhota). Us case me user se photo dobara maango.
Function kabhi exception nahi uchhalta.

## Owner se align karna hai
- Tumhare Learned weights CSV / Product catalogue me body shape ke naam upar wale 5 labels se match hone chahiye.

## Limits
- Sirf photo + gender se shape nikalta hai (height/weight nahi), isliye front+side+height+weight wale model se kam accurate hai. Shape label measurements se bana rule hai, ~30% cases threshold ke paas hote hain.
- Model BodyM (CC BY-NC 4.0, non-commercial) par trained hai. Commercial use se pehle license check karo.
- Photo save nahi hoti; sirf memory me process hoti hai.
