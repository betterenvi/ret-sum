# Code of "A Neural Framework for Retrieval and Summarization of Source Code"

## Setup

TensorFlow version: 1.4

Download the Meteor automatic evaluation tool and put it under the $HOME dir.
```
wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
```

Clone this repo.
```
git clone https://github.com/betterenvi/ret-sum.git
cd ret-sum
mkdir save
```

Start training and evaluation. The evaluation will be performed each training epoch.
```
# retrieve C#
sh ./csharp_ret.sh

# retrieve SQL
sh ./sql_ret.sh

# summarize C#
sh ./csharp_sum.sh

# summarize SQL
sh ./sql_sum.sh
```

Example log line of retrieval task:
```
INFO:model:-------- Result below --------
INFO:root:0.7869 (0.007401) (at 49) 0.7392 (0.007706) 0.7392 (0.007706) (at 49)
INFO:model:-------- Result above --------
```
which means: up to now, we have the best MRR (0.7869 +- 0.007401) at 49th epoch on DEV set; at this epoch, the MRR on EVAL set is 0.7392 +- 0.007706; also at 49th epoch, we have the best Meteor on EVAL set.

Example log line of summarization task:
```
INFO:root:[13.1745, 20.7678] (at 32) [10.6258, 19.4140] [11.1382, 19.8054] (at 19)
```
which means: up to now, we have the best Meteor (13.1745) at 32nd epoch on DEV set; at this epoch, the BLEU on DEV set is 20.7678, and the Meteor and BLEU on EVAL set is 10.6258 and 19.4140; while at the 19th epoch, we have the best Meteor on EVAL set.

Note that since the training process is stochastic (the initialization of weights and the sampling operation in reparameterization), you may get a differnent result (but close).
