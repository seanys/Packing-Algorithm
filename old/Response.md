Thank you for providing careful and valuable comments. We agree that there are significant improvements (including the aspect of writing quality)  to make before it fully fulfill the requirements of CPAIOR. Although we know this paper won't be accepted, we still think we are supposed to explain your concerns.



Reviewers

Q: What is the main issue addressed in this paper?

A: It proved that, based on a stable encode approach, an optimal relative placement of two shapes can be directly found by Neural Networks (NNs). 



Reviewer \#1

Q1: Bad English and poor writing quality

A1: Authors lack academic writing experience and are not good at English (temporarily). We promise that this kind of awful paper which puzzles reviewers will not come from the author of this paper in the future.



Reviewer \#2

Q1: Why networks are a good choice to deal with this problem？

A1: Because NNs can directly obtain a better initial solution for pieces (>8) in nesting problem, which can effectively reduce the run time of optimization steps.



Q2: How does this approach generalize to more than three shapes?

A2: Our approach to encode shapes can handle most shapes. However, in this paper, network training indeed only processed three shapes and we agree that it is insufficient.



Reviewer \#3

Q1: The author never clarify how the proposed NN could be used within a more complex algorithm.

A1: In this paper, we only want to prove that pre-trained NN can be applied to find an optimal relative placement of two shapes. But we agree that it is insufficient.



Q2: The basic shapes chosen for the training sets seem too simple? / Training and test sets are small?

A2: Please refer to A2 to reviewer \#2



Q3: Placement appears to be sometimes infeasible.

A3: It can be optimized by simple methods. For example, translate the shape to the nearest point on No-Fit-Polygon. But this is not within the scope we originally expected to discuss.

We admit that this paper indeed doesn't provide solid enough evidence for NN's effectiveness and we think problems pointed out by reviewer 3 are accurate.



By the way, in our recent experiments, we use NNs to directly find an initial arrangement for pieces (>8) in a containing region and then obtain a feasible solution through minimizing overlap. This process can effectively reduce the run time of optimization steps (shrink&relocate) compared with conventional ways.

Thanks again for your precious comments and especially for review 3's recognition of our idea.
