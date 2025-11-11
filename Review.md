=============================


============================================================================ 
DATE 2026 Reviews for Submission #656
============================================================================ 

Title: HyperAnalog: A Hypergraph Foundation Model for Analog Circuits
Authors: Taejin Paik and Suwan Kim


============================================================================
                            REVIEWER #1
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                           Overall Value: 2
The paper falls within the scope of the topic: Completely

Topic Relevance
---------------------------------------------------------------------------
Thie paper applies advanced AI methods in the form of hypergraph neural networks with transformer enhancements to analog circuit tasks such as classification and specification regression, demonstrating clear practical benefits and fitting the scope of "Applications of Artificial Intelligence Systems."
---------------------------------------------------------------------------


Summary of ideas
---------------------------------------------------------------------------
The paper presents HyperAnalog, a foundation model for analog circuits that uses a transistor-level hypergraph representation with voltage-aware positional encodings and a transformer-enhanced HGNN to capture both structural semantics and long-range dependencies. It achieves 86.1% classification accuracy versus 61.3% for a bipartite baseline, and greatly reduces regression error (e.g., MSE 0.0081 vs. 0.4575 on LNA), demonstrating significant gains in accuracy and generalization for circuit analysis.
---------------------------------------------------------------------------


Strong points
---------------------------------------------------------------------------
- Novel hypergraph representation that models transistor terminal interactions explicitly, preserving structural and electrical semantics lost in conventional graph formulations.
- Voltage-aware sinusoidal positional embeddings that incorporate physical biasing information (distances to VDD/VSS) and enable the model to distinguish structurally similar but functionally different circuits.
- Strong empirical results with significant improvements over baselines, achieving 86.1% classification accuracy vs. 61.3% for bipartite models and large reductions in regression error (e.g., MSE 0.0081 vs. 0.4575 on LNA).
---------------------------------------------------------------------------


Weak points
---------------------------------------------------------------------------
- Limited suitability for generative tasks: Unlike approaches such as AnalogGenie, which leverage Euler-path representations to generate new circuit topologies, HyperAnalog is focused on classification and regression and does not directly support topology generation or exploration.
- Scalability concerns: Hypergraph modeling combined with transformer layers may face challenges in handling very large industrial-scale netlists compared to lighter representations such as Euler paths.
---------------------------------------------------------------------------



============================================================================
                            REVIEWER #2
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                           Overall Value: 1
The paper falls within the scope of the topic: Completely

Summary of ideas
---------------------------------------------------------------------------
This paper proposes HyperAnalog, a hypergraph neural network for analog circuit modelling. The key innovation is a transistor-level hypergraph representation where each transistor is modelled as four nodes representing terminal pairs (gate-source, gate-drain, drain-source, source-body) connected by hyperedges. The architecture combines hypergraph convolution with transformer encoders and incorporates voltage-aware positional encodings based on distances from VDD/VSS power rails. The method is evaluated on circuit topology classification (AnalogGenie dataset, 10 classes) and specification regression (AICircuit benchmark, multiple circuit types).
---------------------------------------------------------------------------


Strong points
---------------------------------------------------------------------------
1. The transistor modelling based on terminal pairs is well-grounded in actual transistor physics (small/large-signal models), providing a principled alternative to arbitrary graph constructions.
2. The need for a native hypergraph approach is well motivated by demonstrating representation ambiguity in simplified graphs (Fig 2a) and semantic inconsistency in heterogeneous representations (Fig 2b). 
3. VDD/VSS positional encodings and the hypergraph structure seem to align with analog circuit properties like voltage biasing and multi-terminal interactions.
4. The proposed solution HyperAnalog seems to addresses both the representation challenge (requirement 1) and the modelling capacity challenge (requirement 2) systematically, and the evaluation results show gains across both classification and regression tasks compared to the bipartite baseline.
---------------------------------------------------------------------------


Weak points
---------------------------------------------------------------------------
1. The evaluation only compares against one baseline (bipartite + Laplacian PE). The paper  criticizes clique expansion methods but provides no empirical comparison against them, making it impossible to validate these criticisms or isolate the value of the hypergraph representation.
2. While the domain-specific adaptations seems effective, the proposed method HyperAnalog mainly relies on a hypergraph GNN with transformer layers and positional encoding, which appears incremental relative to existing HGNN or hybrid models.
3. The use of the term "foundation model" seems to be an over-statement for what is currently a task-specific architecture rather than a pre-trained, generalizable analog modeling system.
---------------------------------------------------------------------------



============================================================================
                            REVIEWER #3
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                           Overall Value: 2
The paper falls within the scope of the topic: Completely

Topic Relevance
---------------------------------------------------------------------------
A new neural representation for analog circuits
---------------------------------------------------------------------------


Summary of ideas
---------------------------------------------------------------------------
This paper proposes a novel hypergraph modeling approach for traditional analog circuits, aiming to resolve the ambiguities inherent in conventional graph representations. Additionally, it introduces a transformer-based Hypergraph Neural Network (HGNN) with positional encoding, designed to enhance the model's ability to differentiate between circuit structures.
---------------------------------------------------------------------------


Strong points
---------------------------------------------------------------------------
+ The proposed combination of hypergraph modeling and a transformer-enhanced HGNN is both novel and effective.
+ The paper is exceptionally well-written. It presents the problem clearly from the introduction, develops the proposed solution logically, and demonstrates its effectiveness through well-structured experiments.
+ The topic is highly relevant to the conference's scope and themes.
---------------------------------------------------------------------------


Weak points
---------------------------------------------------------------------------
- At present, it is not entirely clear what practical applications exist for analog circuit graph representations and classification using neural networks.
---------------------------------------------------------------------------


Other comments
---------------------------------------------------------------------------
Thank you for submitting this excellent work. The paper is well-crafted and accessibleï¼Œ even to me, I am not very familiar with this domain. With a basic understanding of analog circuits, I can also grasp the core ideas and contributions of the paper. The proposed use of hypergraph representation is particularly compelling, offering a more expressive way to model key components such as transistors in analog circuits. The HGNN design also demonstrates strong physical and structural motivation, which enhances its credibility. The evaluation section is well-executed, clearly showcasing the superiority of the proposed HyperAnalog model.

My only major concern lies in the practical utility of neural network-based classification of analog circuits using graph representations. While the work is conceptually interesting and well-grounded, it remains unclear how such methods might be applied in real-world analog design flows. But advancing symbolic representations like these is always valuable and may well play a role in larger, integrated hardware design frameworks in the future.

One minor comment is why existing hypergraph expansion methods, such as star and clique expansion, are not included within the evaluation. Was their exclusion due to expected poor performance, or were there other reasons? A brief justification would improve completeness.
---------------------------------------------------------------------------



============================================================================
                            REVIEWER #4
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                           Overall Value: -1
The paper falls within the scope of the topic: Completely

Topic Relevance
---------------------------------------------------------------------------
Tt directly addresses analog circuit understanding with a transistor-level hypergraph plus Transformer and evaluates on classification and spec-regression tasks.
---------------------------------------------------------------------------


Summary of ideas
---------------------------------------------------------------------------
The paper presents HyperAnalog, an analog-circuit understanding model that decomposes each transistor into terminal-pair nodes (GS, GD, DS, SB) linked by a hyperedge, augments this representation with voltage-aware sinusoidal positional encodings derived from distances to VDD/VSS, and combines hypergraph message passing with a Transformer encoder. Using this setup, the authors report accuracy gains for circuit classification on AnalogGenie and MSE reductions for specification regression on AICircuit when compared with a bipartite representation equipped with Laplacian positional encodings and a Transformer backbone.
---------------------------------------------------------------------------


Strong points
---------------------------------------------------------------------------
The motivation is sensible: explicitly modeling device terminals preserves functional structure that coarse node/edge abstractions can blur. Architecturally, the combination of a hypergraph convolutional stage with attention and a Transformer block is reasonable and practically oriented. Evaluating on both topology-level classification and specification regression addresses two common targets in analog machine learning and demonstrates some breadth.
---------------------------------------------------------------------------


Weak points
---------------------------------------------------------------------------
The case for novelty is not yet convincingly made. Hypergraph formulations for circuits and Transformer-style graph models already exist, so the representational twist of terminal-pair nodes must be validated against strong hypergraph and pin-level alternatives rather than only a bipartite baseline. As it stands, the baseline is too weak for the 2025 landscape. without capacity-matched hypergraph and graph-Transformer comparators, it is hard to separate the benefits of the representation from those of model size or capacity.

The experimental protocol also raises concerns. The manuscript mentions an "80/20 train/test" split and separately refers to a validation set but never defines how that validation set is constructed. Moreover, analog datasets are vulnerable to template or topology leakage, with near-duplicates crossing splits. Without family-disjoint or template-disjoint splits the reported gains may be inflated. 

Metric choices are similarly under-specified: using accuracy alone for classification can be misleading under class imbalance and would benefit from macro-F1, per-class F1, and a confusion matrix. Similarly, relying solely on MSE for regression can hide practical error characteristics, so MAE or RMSE, R2, relative errors normalized by specification ranges, and correlation measures such as Pearson or Spearman would give a clearer picture.

The proposed voltage-aware positional encoding needs sharper definition to rule out leakage. It is not clear whether distances to the power rails are purely structural (graph distances on the netlist) or whether they incorporate simulated operating-point information. If the latter is true and such signals are unavailable at inference time, the method would be leaking information. This must be clarified and ablated with and without any simulation-derived quantities (I might have misunderstood this part and no simulations are involved, but then I wonder how do you get the voltage information. In any case, please make it clearer). 

The physical fidelity of the transistor decomposition also merits justification. While GS, GD, DS, and SB nodes are modeled, GB and DB pairs are omitted. I agree that GB is often negligible, but DB junction capacitance is a standard high-frequency element. At minimum, the omission should be justified by operating region and frequency, and the work would be complete if an ablation study that adds GB/DB demonstrates that they are unnecessary for the reported tasks.

Training details are too sparse to be reproducible or to allow fair comparison. The paper mentions a small learning rate with cosine decay, batch size 32, and gradient clipping at 0.5, but omits the actual learning-rate value, warmup strategy, and hyperparameters, weight decay, number of layers and heads, hidden dimensions, dropout, and the hyperparameter search space and budget for both the baseline and HyperAnalog. 

Learning curves â€” training and test loss versus steps â€” are especially important here and should be provided, along with a clear explanation of how the validation set is defined if the split remains 80/20.

Finally, the ablation study is not yet sufficient to localize the source of gains. Removing only the positional encodings and the reweighting leaves open whether the hypergraph stage, the Transformer stage, the anchoring features on hyperedges, or the representation itself is doing the heavy lifting. The ablations should include a model without the Transformer (hypergraph convolution only), a Transformer-only variant without the hypergraph convolution, a version without hyperedge anchor features, a representation swap contrasting transistor-as-hyperedge, pin-level graphs, and the proposed terminal-pair nodes, terminal-pair ablations that remove SB or GD or add GB/DB, and a test of directionality and device symmetry (such as source/drain swaps). 

Beyond this, the work does not yet demonstrate generalization: there are no out-of-distribution tests on unseen topologies or device counts, no cross-dataset experiments, and no scaling study.

In terms of positioning, the paper argues primarily against bipartite graphs and thereby undersells the relevant landscape. A fuller treatment would discuss and compare with directed hypergraph GNNs, Hypergraph Transformers, Graphormer-style graph Transformers, and pin-level circuit representations. It should also make explicit what is genuinely new: the terminal-pair nodeization, the particular fusion of hypergraph convolution and Transformer, and the voltage-aware sinusoidal encoding, and isolate each of these contributions in ablations to substantiate the claims.
---------------------------------------------------------------------------
