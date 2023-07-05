Directory that contains raw data cubes. Cubes are assumed to have axes in order "xyz", for instance:

```
NAXIS   =                    3         / 3 axes
NAXIS1  =                 1074         / Number of pixels along the x axis
NAXIS2  =                  758         / Number of pixels along the y axis
NAXIS3  =                  240         / Number of velocity channels
```

If your cubes doe not match this convention, consider swapping axes before processing them.
