
# Exercise 66 - Test Evidence

## Input
Red: [255, 0, 0]
Blue: [0, 0, 255]
Red: [255, 0, 0]

## Expected result
2 unique RGB colours.

## Actual result
2 unique RGB colours.

## Intermediate values
- Shape: (1, 3, 3)
- Dtype: uint8
- Standard method: 2
- Faster method: 2

## Verification
- RGB channel order: PASS
- uint8 dtype: PASS
- Unique colour count: PASS
- Incorrect result detection: PASS

## Checker
python work/R44-T041/check.py
