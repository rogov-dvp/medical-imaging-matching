# Regression Testing Instructions

## Idea of regression testing
Regression testing is used to check if any updates changed the values before the updates. ie. regression test does not check if a function or system is producing correct results, but rather if the code change made any functions return a different value than before the code change.

### installation
https://pypi.org/project/pytest-regtest/

pytest-regtest 1.4.6
`pip install pytest-regtest`

### CLI
1. Set up or reset your "correct" answers with this CLI:
`pytest --regtest-reset <filename>`

2. After code change, check for any changes with this CLI:
`pytest <filename>` 

Note, when running the CLI, dots mean there is no change and everything is working as intended. An F, means a failure or change in code.

### Examples

```
import cv2
from ORB import orb_sim, img1, img2

def test_general(regtest):
    value = orb_sim(img1, img2)
    print(value,file=regtest)   #Write the regression test into file.

```

```
import cv2
from SSIM import img1

# Check if img1 exists (file path is correct)
def test_img1_not_none(regtest):
    print(type(img1) != None,file=regtest)
```
