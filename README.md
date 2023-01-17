# How does "softmax_cross_entropy_with_logits" work ?
## Maths behind:
### Step - 01:
Calculate softmax of logits using equation

#### f(s) = e^s/∑e^s

Here, s is logit

### Step - 02:

Then Calculate Cross Entropy Loss:

#### CE = - ∑ t * log(f(s)) [t is label]
