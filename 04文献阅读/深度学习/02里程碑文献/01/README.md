# Deep learning2015Nature
1 Facebook AI 研究院，770 Broadway，纽约，纽约州 10003 美国。
2 纽约大学，715 Broadway，纽约，纽约州 10003，美国。
3 蒙特利尔大学计算机科学与运筹学系，安德烈-艾森斯塔特楼，邮政信箱 6128 中心站，蒙特利尔，魁北克省 H3C 3J7，加拿大。
4 Google，1600 Amphitheatre Parkway，山景城，加利福尼亚州 94043，美国。
5 多伦多大学计算机科学系，
6 King's College Road，多伦多，安大略省 M5S 3G4，加拿大。

机器学习技术为现代社会的许多方面提供动力：从网络搜索到社交网络上的内容过滤，再到电子商务网站上的推荐，它越来越多地出现在消费产品如相机和智能手机中。
机器学习系统被用来识别图像中的对象，将语音转录成文本，匹配新闻项目、帖子或产品与用户的兴趣，并选择搜索的相关结果。
越来越多地，这些应用程序利用了一类称为深度学习的技术。

传统机器学习技术在处理原始形式的自然数据方面的能力是有限的。
几十年来，构建一个模式识别或机器学习系统需要精心设计和相当的领域专业知识来设计一个特征提取器，该提取器将原始数据（如图像的像素值）转换为适合的内部表示或特征向量，从这些向量中，学习子系统（通常是分类器）可以检测或分类输入中的模式。

表示学习是一组方法，允许机器被输入原始数据，并自动发现检测或分类所需的表示。
深度学习方法是通过将简单的非线性模块组合起来，每个模块都将一个层次的表示（从原始输入开始）转换为更高、稍微更抽象层次的表示，从而获得多层次表示的表示学习方法。通过足够多的这种转换的组合，可以学习到非常复杂的函数。对于分类任务，表示的更高层次放大了输入中对区分重要的方面，并抑制了不相关的变异。例如，图像以像素值数组的形式进入，表示的第一层通常表示图像中特定方向和位置的边缘的存在或不存在。第二层通常通过发现边缘的特定排列来检测图案，而不管边缘位置的小变化。第三层可能将图案组装成与熟悉对象的部分相对应的较大组合，后续层次将检测对象作为这些部分的组合。深度学习的关键方面是这些特征层不是由人类工程师设计的：它们是从数据中使用通用学习过程学习的。

深度学习在解决人工智能社区多年来最好的尝试都未能解决的问题方面取得了重大进展。事实证明，它非常擅长发现高维数据中的复杂结构，因此适用于科学、商业和政府的许多领域。除了在图像识别1-4和语音识别5-7中打破记录外，它还在预测潜在药物分子的活性8、分析粒子加速器数据9,10、重建脑回路11以及预测非编码DNA中突变对基因表达和疾病的影响12,13方面超越了其他机器学习技术。也许更令人惊讶的是，深度学习在自然语言理解的各种任务中取得了极为有希望的结果14，特别是主题分类、情感分析、问题回答15和语言翻译16,17。

我们认为，深度学习将在不久的将来取得更多的成功，因为它几乎不需要手工工程，因此可以轻松地利用可用计算和数据量的增加。目前正在为深度神经网络开发的新学习算法和架构将加速这一进展。

监督学习 最常见的机器学习形式，无论是深度还是非深度，都是监督学习。想象一下我们想要构建一个能够将图像分类为包含，比如说，房子、汽车、人或宠物的系统。我们首先收集大量标记有类别的房屋、汽车、人和宠物的图像数据集。在训练期间，机器被展示一张图像，并以每个类别一个分数向量的输出形式产生输出。我们希望所需类别的得分是所有类别中最高的，但在训练之前这不太可能发生。我们计算一个目标函数，衡量输出分数与期望分数模式之间的误差（或距离）。然后，机器修改其内部可调参数以减少这种误差。这些可调参数，通常称为权重，是可以看作定义机器输入输出功能的“旋钮”的实数。在典型的深度学习系统中，可能有数亿个这样的可调权重，以及数亿个标记示例来训练机器。

为了正确调整权重向量，学习算法计算一个梯度向量，对于每个权重，它指示如果权重增加一个微小量，误差会增加或减少的量。然后，权重向量朝与梯度向量相反的方向进行调整。

目标函数在所有训练示例上平均后，可以被视为权重值的高维空间中的某种丘陵景观。负梯度向量指示这个景观中陡峭下降的方向，将其带向一个最小值，那里的平均输出误差较低。

在实践中，大多数从业者使用一种称为随机梯度下降（SGD）的程序。这包括展示几个示例的输入向量，计算输出和误差，计算这些示例的平均梯度，并相应地调整权重。这个过程在训练集的许多小示例集上重复进行，直到目标函数的平均值停止下降。它被称为随机的，因为每个小示例集给出了所有示例的平均梯度的嘈杂估计。这种简单的过程通常在与远更复杂的优化技术相比时，能够令人惊讶地快速找到一组好的权重18。训练后，系统的性能在称为测试集的不同示例集上进行测量。这用于测试机器的泛化能力——它在从未在训练期间见过的新输入上产生合理答案的能力。

目前机器学习的实际应用中，许多使用手工设计的特征上的线性分类器。一个二类线性分类器计算特征向量分量的加权和。如果加权和高于阈值，输入就被分类为属于特定类别。

自20世纪60年代以来，我们知道线性分类器只能将输入空间划分为非常简单的区域，即由超平面分隔的半空间19。但是，像图像和语音识别这样的问题要求输入输出函数对输入的不相关变化不敏感，例如对象的位置、方向或照明的变化，或者语音的音高或口音的变化，同时对特定的微小变化非常敏感（例如，白狼和称为萨摩耶德的类似白狼的犬种之间的区别）。在像素级别，两个不同姿势和不同环境中的萨摩耶德的图像可能彼此非常不同，而同一位置和类似背景下的萨摩耶德和狼的两个图像可能彼此非常相似。线性分类器，或任何其他在原始像素上操作的“浅层”分类器，不可能区分后者两者，而将前者两者归为同一类别。这就是为什么浅层分类器需要一个好的特征提取器来解决选择性-不变性困境——一个产生对图像的某些方面有选择性但对动物的姿势等不相关方面不变的表示。为了使分类器更强大，可以使用通用的非线性特征，如与核方法20一起使用的那样，但像高斯核这样的通用特征不允许学习者远离训练示例很好地泛化21。传统选择是手工设计好的特征提取器，这需要相当多的工程技能和领域专业知识。但如果可以使用通用的学习过程自动学习好的特征，所有这些都可以避免。这是深度学习的关键优势。

深度学习架构是一个简单模块的多层堆栈，所有（或大多数）模块都受到学习的影响，许多模块计算非线性输入-输出映射。堆栈中的每个模块都转换其输入以增加表示的选择性和不变性。通过多个非线性层，比如说5到20层的深度，一个系统可以实施其输入的极其复杂的函数，同时对微小的细节敏感——区分萨摩耶德和白狼——并对大的不相关变化如背景、姿势、照明和周围对象不敏感。

反向传播训练多层架构 从模式识别的最早日子开始22,23，研究人员的目标就是用可训练的多层网络取代手工设计的特征，尽管它的解决方案很简单，直到1980年代中期才被广泛理解。事实证明，多层架构可以通过简单的随机梯度下降进行训练。只要模块是其输入和内部权重的相对平滑函数，就可以使用反向传播过程计算梯度。这个想法可以在1970年代和1980年代由几个不同的研究小组独立发现24-27。

反向传播过程计算一个多层模块堆栈的权重的目标函数梯度，只不过是梯度链式法则的实用应用。关键的洞察是，通过从该模块的输出（或后续模块的输入）的梯度向后工作，可以计算出目标相对于模块输入的导数（或梯度）（图1）。反向传播方程可以反复应用，将梯度通过所有模块传播，从顶部的输出开始（网络在那里产生其预测），一直到底部（外部输入被输入）。一旦计算出这些梯度，就可以直接计算出每个模块权重的梯度。

许多深度学习的应用使用前馈神经网络架构（图1），它们学习将固定大小的输入（例如，一张图像）映射到固定大小的输出（例如，几个类别的概率）。为了从一个层次到下一个层次，一组单元计算其来自前一层次的输入的加权和，并将结果通过非线性函数传递。目前，最受欢迎的非线性函数是修正线性单元（ReLU），它简单地是半波整流器f(z) = max(z, 0)。在过去的几十年中，神经网络使用了更平滑的非线性，例如tanh(z)或1/(1 + exp(−z))，但ReLU通常在许多层的网络中学习得更快，允许在没有无监督预训练的情况下训练深度监督网络28。不在输入或输出层的单元通常被称为隐藏单元。隐藏层可以看作是以非线性方式扭曲输入，使得类别在最后一层变得线性可分（图1）。

在20世纪90年代末，神经网络和反向传播在很大程度上被机器学习社区抛弃，并且被计算机视觉和语音识别社区忽视。人们普遍认为，用很少的先验知识学习有用的多阶段特征提取器是不可行的。特别是，人们普遍认为简单的梯度下降会陷入较差的局部最小值——权重配置，对于这些配置，没有小的变化会降低平均误差。

在实践中，对于大型网络来说，较差的局部最小值很少是问题。无论初始条件如何，系统几乎总是达到非常相似质量的解决方案。最近的理论和经验结果强烈表明，局部最小值通常不是一般问题。相反，这个景观充满了组合上的大量鞍点，其中梯度为零，表面在大多数维度上向上弯曲，在其余维度上向下弯曲29,30。分析似乎表明，只有少数向下弯曲方向的鞍点在非常大的数量中存在，但几乎所有这些鞍点的目标函数值都非常相似。因此，算法陷入这些鞍点中的哪一个并不重要。

对深度前馈网络的兴趣在2006年左右被重新唤起，这要归功于加拿大高级研究所（CIFAR）聚集的一组研究人员。研究人员引入了无监督学习程序，可以在不需要标记数据的情况下创建特征检测器层。学习每一层特征检测器的目标是能够重建或模拟下面一层的特征检测器（或原始输入）的活动。通过使用这种重建目标“预训练”几个逐渐更复杂的特征检测器层，可以将深度网络的权重初始化为合理的值。然后可以在网络的顶部添加一个输出单元的最终层，并使用标准反向传播对整个深度系统进行微调33-35。这对于识别手写数字或检测行人非常有效，特别是当标记数据量非常有限时36。

这种预训练方法的首次重大应用是在语音识别中，这得益于快速图形处理单元（GPU）的出现，这些GPU易于编程37，并允许研究人员以比以往快10或20倍的速度训练网络。2009年，该方法被用于将从声波中提取的系数的短时间窗口映射到可能由窗口中心的帧表示的语音片段的各种片段的概率集。它在一个小词汇量的标准语音识别基准测试中取得了突破性的结果38，并迅速发展到在一个大词汇量任务上取得突破性的结果39。到2012年，2009年的深度网络版本正在被许多主要的语音团队开发6，并已经部署在Android手机中。对于较小的数据集，无监督预训练有助于防止过拟合40，从而在标记示例数量较少时，或在转移设置中，我们有许多“源”任务的示例但只有很少的“目标”任务的示例时，导致显著更好的泛化。一旦深度学习得到复兴，事实证明，预训练阶段只需要用于较小的数据集。

然而，有一种特定类型的深度前馈网络比具有相邻层之间完全连接的网络更容易训练，并且泛化效果更好。这就是卷积神经网络（ConvNet）41,42。它在神经网络不受欢迎的时期取得了许多实际成功，并最近被计算机视觉社区广泛采用。

卷积神经网络 ConvNets旨在处理以多个数组形式出现的数据，例如由三个包含三个颜色通道像素强度的2D数组组成的彩色图像。许多数据模态都是以多个数组的形式存在的：1D用于信号和序列，包括语言；2D用于图像或音频频谱图；3D用于视频或体积图像。ConvNets背后的四个关键思想利用了自然信号的特性：局部连接、共享权重、池化和使用多层。

典型的ConvNet的架构（图2）是由一系列阶段组成的。前几个阶段由两种类型的层组成：卷积层和池化层。卷积层中的单元组织在特征图中，其中每个单元通过称为滤波器库的一组权重与前一层特征图的局部补丁相连。这个局部加权和的结果然后通过ReLU等非线性函数传递。一个特征图中的所有单元共享相同的滤波器库。层中的不同特征图使用不同的滤波器库。这种架构的原因有两个。首先，在图像等数组数据中，局部值组通常是高度相关的，形成容易检测的独特局部图案。其次，图像和其他信号的局部统计特性对位置不变。换句话说，如果一个图案可以出现在图像的一个部分，它可能无处不在，因此不同位置的单元共享相同的权重并在数组的不同部分检测相同的图案。数学上，特征图执行的过滤操作是离散卷积，因此得名。

尽管卷积层的作用是检测前一层的局部特征组合，但池化层的作用是将语义上相似的特征合并为一个。因为形成图案的特征的相对位置可能会有所变化，通过粗化每个特征的位置来可靠地检测图案。典型的池化单元计算一个特征图中局部补丁单元的最大值（或在少数特征图中）。邻近的池化单元从被移位了一个或多个行或列的补丁中获取输入，从而降低了表示的维度，并创建了对小的位移和扭曲的不变性。两到三个卷积、非线性和池化阶段堆叠在一起，然后是更多的卷积和全连接层。通过ConvNet反向传播梯度就像通过常规深度网络一样简单，允许所有滤波器库中的所有权重进行训练。

深度神经网络利用了许多自然信号是组合层次结构的特性，其中更高层次的特征是通过组合较低层次的特征获得的。在图像中，边缘的局部组合形成图案，图案组装成部分，部分形成对象。在从声音到音素、音节、单词和句子的语音和文本中存在类似的层次结构。池化允许表示在前一层的元素在位置和外观变化时变化非常小。

ConvNets中的卷积和池化层直接受到视觉神经科学中经典简单细胞和复杂细胞概念的启发43，整体架构让人想起视觉皮层腹侧通路中的LGN-V1-V2-V4-IT层次结构44。当ConvNet模型和猴子被展示相同的图片时，ConvNet中高级单元的激活解释了猴子颞下皮层中随机160个神经元的方差的一半45。ConvNets的根源在于neocognitron46，其架构有些相似，但没有像反向传播这样的端到端监督学习算法。一个原始的1D ConvNet称为时延神经网络被用于识别音素和简单单词47,48。

从20世纪90年代初开始，卷积网络的应用就已经非常广泛，从用于语音识别47和文档阅读42的时延神经网络开始。文档阅读系统使用了一个与实现语言约束的概率模型联合训练的ConvNet。到20世纪90年代末，这个系统已经阅读了美国超过10%的支票。微软后来部署了一些基于ConvNet的光学字符识别和手写识别系统49。ConvNets在20世纪90年代初也被用于自然图像中的对象检测，包括面部和手部50,51，以及面部识别52。

自21世纪初以来，ConvNets已经成功地应用于图像中对象和区域的检测、分割和识别。这些都是标记数据相对丰富的任务，如交通标志识别53，生物图像的分割54，特别是用于连接组学55，以及在自然图像中检测面部、文本、行人和人体36,50,51,56-58。ConvNets的一个重大实际成功是面部识别59。

重要的是，图像可以在像素级别进行标记，这将在技术中有所应用，包括自主移动机器人和自动驾驶汽车60,61。Mobileye和NVIDIA等公司正在其即将推出的汽车视觉系统中使用基于ConvNet的方法。其他日益重要的应用涉及自然语言理解和语音识别7。

尽管取得了这些成功，但直到2012年ImageNet竞赛，ConvNets在主流计算机视觉和机器学习社区中基本上被抛弃。当深度卷积网络应用于包含1000个不同类别的约一百万张来自网络的图像数据集时，它们取得了惊人的结果，几乎将最佳竞争方法的错误率减半1。这一成功来自于有效使用GPU、ReLU、一种称为dropout的新正则化技术62，以及通过变形现有图像来生成更多训练示例的技术。这一成功引发了计算机视觉领域的革命；ConvNets现在是几乎所有识别和检测任务的主导方法4,58,59,63-65，并且在某些任务上接近人类的表现。最近的一项令人惊叹的演示结合了ConvNets和循环网络模块，用于生成图像字幕（图3）。

最近的ConvNet架构具有10到20层的ReLU，数亿个权重，以及数十亿个单元之间的连接。虽然训练如此大规模的网络可能只需要几周的时间，但硬件、软件和算法并行化的进步已经将训练时间缩短到几个小时。

ConvNet基础视觉系统的性能导致大多数主要技术公司，包括Google、Facebook、Microsoft、IBM、Yahoo!、Twitter和Adobe，以及越来越多的初创公司，启动研究和开发项目，并部署基于ConvNet的图像理解产品和服务。ConvNets易于适应在芯片或现场可编程门阵列（FPGA）中的高效硬件实现66,67。一些公司，如NVIDIA、Mobileye、Intel、Qualcomm和Samsung，正在开发ConvNet芯片，以实现智能手机、相机、机器人和自动驾驶汽车中的实时视觉应用。

分布式表示和语言处理 深度学习理论表明，深度网络在不使用分布式表示的经典学习算法方面具有两个不同的指数优势21。这两个优势都源于组合的力量，并且依赖于底层数据生成分布具有适当的成分结构40。首先，学习分布式表示使得能够泛化到训练期间未见过的学习特征值的新组合（例如，具有n个二进制特征时可能存在2^n种组合）68,69。其次，在深度网络中组合表示层带来了另一个指数优势的潜力70（指数与深度相关）。

多层神经网络的隐藏层学习以一种方式表示网络的输入，使得预测目标输出变得容易。通过训练多层神经网络来预测序列中的下一个单词，这一点得到了很好的证明，从局部上下文71中。在第一层，每个单词创建不同的激活模式，或单词向量（图4）。在语言模型中，网络的其他层学习将输入单词向量转换为预测下一个单词的输出单词向量，该向量可用于预测词汇表中任何单词作为下一个单词出现的概率。网络学习到的单词向量包含许多活动组件，每个组件都可以解释为单词的一个独立特征，正如27在符号的分布式表示学习的背景下首次展示的那样。这些语义特征在输入中并没有明确存在。它们是通过学习过程发现的，作为一种良好的方式，将输入和输出符号之间的结构化关系分解为多个“微观规则”。当单词序列来自大量真实文本，并且个别微观规则不可靠时，学习单词向量也被证明非常有效71。例如，当训练用于预测新闻故事中的下一个单词时，Tuesday和Wednesday的学习到的单词向量非常相似，Sweden和Norway的单词向量也是如此。这种表示被称为分布式表示，因为它们的元素（特征）不是相互排斥的，它们的许多配置对应于观察到的数据变化。这些单词向量由神经网络自动发现的学习特征组成。从文本中学习的单词的向量表示现在在自然语言应用14,17,72-76中被广泛使用。

表示问题处于逻辑启发和神经网络启发认知范式之间争论的核心。在逻辑启发的范式中，符号的一个实例是某物，其唯一属性是它与其他符号实例相同或不同。它没有与其使用相关的内部结构；并且要对符号进行推理，它们必须绑定到精心选择的推理规则中的变量。相比之下，神经网络只是使用大型活动向量、大型权重矩阵和标量非线性来执行快速的“直观”推理，这种推理构成了轻松常识推理的基础。

在引入神经语言模型71之前，统计语言建模的标准方法没有利用分布式表示：它基于计算长度高达N的短符号序列的出现频率（称为N-gram）。可能的N-gram数量在V^N的数量级，其中V是词汇表大小，因此考虑超过少数几个单词的上下文将需要非常大的训练语料库。N-gram将每个单词视为原子单元，因此它们不能泛化到语义相关的单词序列，而神经语言模型可以，因为它们将每个单词与实值特征向量相关联，并且语义相关的单词在该向量空间中最终靠近彼此（图4）。

循环神经网络 当反向传播首次引入时，它最令人兴奋的用途是用于训练循环神经网络（RNNs）。对于涉及序列输入的任务，如语音和语言，通常最好使用RNNs（图5）。RNNs一次处理输入序列的一个元素，在其隐藏单元中保持一个“状态向量”，该向量隐含地包含有关序列所有过去元素的历史信息。当我们考虑不同离散时间步的隐藏单元的输出，就好像它们是深度多层网络中不同神经元的输出一样（图5，右），就变得清楚了，我们如何可以将反向传播应用于训练RNNs。

RNNs是非常强大的动态系统，但训练它们已被证明是有问题的，因为反向传播的梯度在每个时间步要么增长要么缩小，所以经过许多时间步后，它们通常会爆炸或消失77,78。

由于其架构79,80的进步和训练方法81,82的发展，RNNs被发现非常擅长预测文本中的下一个字符83或序列中的下一个单词75，但它们也可以用于更复杂的任务。例如，逐个单词阅读英文句子后，可以训练一个英文“编码器”网络，使其隐藏单元的最终状态向量是句子表达的思想的良好表示。然后，这个思想向量可以用作（或作为额外输入提供给）一个共同训练的法文“解码器”网络的初始隐藏状态，该网络输出法文翻译的第一个单词的概率分布。如果从这个分布中选择一个特定的第一个单词并将其提供给解码器网络作为输入，它将输出翻译的第二个单词的概率分布，依此类推，直到选择一个句号17,72,76。总的来说，这个过程根据依赖于英文句子的概率分布生成一系列法文单词。这种进行机器翻译的相当天真的方式已经迅速与最先进的技术竞争，这引起了人们对是否需要使用推理规则操作的内部符号表达来进行句子理解的严重怀疑。它更符合这样的观点：日常推理涉及许多同时进行的类比84,85。

除了将法文句子的意义翻译成英文句子之外，人们还可以学习将图像的意义“翻译”成英文句子（图3）。这里的编码器是一个深度ConvNet，它将像素转换为其最后一个隐藏层中的活动向量。解码器是一个类似于用于机器翻译和神经语言建模的RNN。最近对这种系统的兴趣激增（参见参考文献86中提到的例子）。

一旦在时间上展开（图5），RNNs可以被视为非常深的前馈网络，其中所有层共享相同的权重。尽管它们的主要用途是学习长期依赖关系，但理论和经验证据表明，学习存储非常长时间的信息是困难的78。

为了纠正这一点，一个想法是通过网络增加一个显式记忆。这种类型的第一种提议是长短期记忆（LSTM）网络，它们使用特殊的隐藏单元，其自然行为是长时间记住输入79。一个名为记忆单元的特殊单元就像一个累加器或门控泄漏神经元：它在下一个时间步有一个连接到自己的连接，权重为一，因此它复制自己的实值状态并累积外部信号，但这个自连接被另一个单元乘法门控，该单元学习决定何时清除记忆的内容。

LSTM网络随后被证明比传统的RNNs更有效，特别是当它们对每个时间步有几个层次时87，使得整个语音识别系统从声学到转录中字符序列的全部过程成为可能。LSTM网络或相关形式的门控单元目前也用于执行机器翻译的编码器和解码器网络，这些网络在性能上表现出色17,72,76。

在过去的一年里，几位作者提出了不同的提议，以增加RNNs的内存模块。提议包括神经图灵机，网络通过“磁带状”内存增强，RNN可以选择从中读取或写入88，以及记忆网络，其中常规网络通过某种联想记忆增强89。记忆网络在标准问答基准测试上取得了优异的性能。内存用于记住网络后来被要求回答问题的故事。

超越简单的记忆，神经图灵机和记忆网络被用于通常需要推理和符号操作的任务。神经图灵机可以被教导“算法”。除其他事项外，它们可以学会输出一个符号的排序列表，当它们的输入包含一个未排序的序列时，其中每个符号都伴随着一个实值，表示它在列表中的优先级88。记忆网络可以被训练以跟踪类似于文本冒险游戏的世界状态，在阅读故事后，它们可以回答需要复杂推理的问题90。在一个测试示例中，网络被展示了《指环王》的15句版本，并正确回答了“弗罗多现在在哪里？”这样的问题89。

深度学习的未来 无监督学习91-98在重新激发对深度学习的兴趣方面起到了催化作用，但此后被纯粹监督学习的成功所掩盖。虽然我们在这篇综述中没有关注它，但我们预计无监督学习在长期内将变得更加重要。人类和动物的学习在很大程度上是无监督的：我们通过观察世界来发现其结构，而不是被告知每个对象的名称。

人类视觉是一个主动的过程，它使用一个小的、高分辨率的中央凹以智能的、特定任务的方式顺序采样光阵列，周围是一个大的、低分辨率的环绕。我们预计，未来的大部分视觉进展将来自于系统，这些系统是端到端训练的，并结合了使用强化学习决定在哪里看的ConvNets和RNNs。结合深度学习和强化学习的系统还处于起步阶段，但它们已经在分类任务上超越了被动视觉系统99，并在学习玩许多不同的视频游戏方面取得了令人印象深刻的结果100。

自然语言理解是深度学习预计在未来几年将产生巨大影响的另一个领域。我们预计，使用RNNs理解句子或整个文档的系统将在它们学习选择性关注一部分的策略时变得更好76,86。

最终，人工智能的重大进展将来自于结合表示学习和复杂推理的系统。尽管深度学习和简单的推理已经用于语音和手写识别很长时间了，但需要新的范式来取代基于规则的符号表达式操作，通过在大型向量上的操作来实现101。■

收到日期：2015年2月25日；接受日期：2015年5月1日。

1. Krizhevsky, A., Sutskever, I. & Hinton, G. ImageNet classification with deep convolutional neural networks. In Proc. Advances in Neural Information Processing Systems 25 1090–1098 (2012).
这份报告是一个突破，使用卷积网络将对象识别的错误率几乎减半，并促使计算机视觉社区迅速采用深度学习。

1. Farabet, C., Couprie, C., Najman, L. & LeCun, Y. Learning hierarchical features for scene labeling. IEEE Trans. Pattern Anal. Mach. Intell. 35, 1915–1929 (2013).

2. Tompson, J., Jain, A., LeCun, Y. & Bregler, C. Joint training of a convolutional network and a graphical model for human pose estimation. In Proc. Advances in Neural Information Processing Systems 27 1799–1807 (2014).

3. Szegedy, C. et al. Going deeper with convolutions. Preprint at http://arxiv.org/abs/1409.4842 (2014).

4. Mikolov, T., Deoras, A., Povey, D., Burget, L. & Cernocky, J. Strategies for training large scale neural network language models. In Proc. Automatic Speech Recognition and Understanding 196–201 (2011).

5. Hinton, G. et al. Deep neural networks for acoustic modeling in speech recognition. IEEE Signal Processing Magazine 29, 82–97 (2012).
这篇来自主要语音识别实验室的联合论文，总结了在自动语音识别的音素分类任务中使用深度学习取得的突破，是深度学习的首个主要工业应用。

1. Sainath, T., Mohamed, A.-R., Kingsbury, B. & Ramabhadran, B. Deep convolutional neural networks for LVCSR. In Proc. Acoustics, Speech and Signal Processing 8614–8618 (2013).

2. Ma, J., Sheridan, R. P., Liaw, A., Dahl, G. E. & Svetnik, V. Deep neural nets as a method for quantitative structure-activity relationships. J. Chem. Inf. Model. 55, 263–274 (2015).

3. Ciodaro, T., Deva, D., de Seixas, J. & Damazio, D. Online particle detection with neural networks based on topological calorimetry information. J. Phys. Conf. Series 368, 012030 (2012).

4.  Kaggle. Higgs boson machine learning challenge. Kaggle https://www.kaggle.com/c/higgs-boson (2014).

5.  Helmstaedter, M. et al. Connectomic reconstruction of the inner plexiform layer in the mouse retina. Nature 500, 168–174 (2013).

6.  Leung, M. K., Xiong, H. Y., Lee, L. J. & Frey, B. J. Deep learning of the tissue regulated splicing code. Bioinformatics 30, i121–i129 (2014).

7.  Xiong, H. Y. et al. The human splicing code reveals new insights into the genetic determinants of disease. Science 347, 6218 (2015).

8.  Collobert, R., et al. Natural language processing (almost) from scratch. J. Mach. Learn. Res. 12, 2493–2537 (2011).

9.  Bordes, A., Chopra, S. & Weston, J. Question answering with subgraph embeddings. In Proc. Empirical Methods in Natural Language Processing http://arxiv.org/abs/1406.3676v3 (2014).

10. Jean, S., Cho, K., Memisevic, R. & Bengio, Y. On using very large target vocabulary for neural machine translation. In Proc. ACL-IJCNLP http://arxiv.org/abs/1412.2007 (2015).

11. Sutskever, I. Vinyals, O. & Le. Q. V. Sequence to sequence learning with neural networks. In Proc. Advances in Neural Information Processing Systems 27 3104–3112 (2014).
这篇论文展示了使用在参考文献72中介绍的架构的最新机器翻译结果，其中一个循环网络被训练来阅读一种语言的句子，产生其含义的语义表示，并生成另一种语言的翻译。

1.  Bottou, L. & Bousquet, O. The tradeoffs of large scale learning. In Proc. Advances in Neural Information Processing Systems 20 161–168 (2007).

2.  Duda, R. O. & Hart, P. E. Pattern Classiﬁcation and Scene Analysis (Wiley, 1973).

3.  Schölkopf, B. & Smola, A. Learning with Kernels (MIT Press, 2002).

4.  Bengio, Y., Delalleau, O. & Le Roux, N. The curse of highly variable functions for local kernel machines. In Proc. Advances in Neural Information Processing Systems 18 107–114 (2005).

5.  Selfridge, O. G. Pandemonium: a paradigm for learning in mechanisation of thought processes. In Proc. Symposium on Mechanisation of Thought Processes 513–526 (1958).

6.  Rosenblatt, F. The Perceptron — A Perceiving and Recognizing Automaton. Tech. Rep. 85-460-1 (Cornell Aeronautical Laboratory, 1957).

7.  Werbos, P. Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences. PhD thesis, Harvard Univ. (1974).

8.  Parker, D. B. Learning Logic Report TR–47 (MIT Press, 1985).

9.  LeCun, Y. Une procédure d’apprentissage pour Réseau à seuil assymétrique in Cognitiva 85: a la Frontière de l’Intelligence Artiﬁcielle, des Sciences de la Connaissance et des Neurosciences [in French] 599–604 (1985).

10. Rumelhart, D. E., Hinton, G. E. & Williams, R. J. Learning representations by back-propagating errors. Nature 323, 533–536 (1986).

11. Glorot, X., Bordes, A. & Bengio. Y. Deep sparse rectiﬁer neural networks. In Proc. 14th International Conference on Artificial Intelligence and Statistics 315–323 (2011).
这篇论文展示了如果隐藏层由ReLU组成，那么非常深的神经网络的监督训练会快得多。

1.  Dauphin, Y. et al. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. In Proc. Advances in Neural Information Processing Systems 27 2933–2941 (2014).

2.  Choromanska, A., Henaff, M., Mathieu, M., Arous, G. B. & LeCun, Y. The loss surface of multilayer networks. In Proc. Conference on AI and Statistics http://arxiv.org/abs/1412.0233 (2014).

3.  Hinton, G. E. What kind of graphical model is the brain? In Proc. 19th International Joint Conference on Artificial intelligence 1765–1775 (2005).

4.  Hinton, G. E., Osindero, S. & Teh, Y.-W. A fast learning algorithm for deep belief nets. Neural Comp. 18, 1527–1554 (2006).
这篇论文介绍了一种新颖有效的方法，通过使用受限玻尔兹曼机的无监督学习过程逐层预训练一个隐藏层，来训练非常深的神经网络。

1.  Bengio, Y., Lamblin, P., Popovici, D. & Larochelle, H. Greedy layer-wise training of deep networks. In Proc. Advances in Neural Information Processing Systems 19 153–160 (2006).
这篇报告证明了参考文献32中介绍的无监督预训练方法显著提高了测试数据上的性能，并将该方法推广到其他无监督表示学习方法，如自编码器。

1.  Ranzato, M., Poultney, C., Chopra, S. & LeCun, Y. Efficient learning of sparse representations with an energy-based model. In Proc. Advances in Neural Information Processing Systems 19 1137–1144 (2006).

2.  Hinton, G. E. & Salakhutdinov, R. Reducing the dimensionality of data with neural networks. Science 313, 504–507 (2006).

3.  Sermanet, P., Kavukcuoglu, K., Chintala, S. & LeCun, Y. Pedestrian detection with unsupervised multi-stage feature learning. In Proc. International Conference on Computer Vision and Pattern Raina, R., Madhavan, A. & Ng, A. Y. Large-scale deep unsupervised learning using graphics processors. In Proc. 26th Annual International Conference on Machine Learning 873–880 (2009).
这篇论文介绍了使用图形处理器进行大规模无监督学习的方法。

1.   Mohamed, A.-R., Dahl, G. E. & Hinton, G. Acoustic modeling using deep belief networks. IEEE Trans. Audio Speech Lang. Process. 20, 14–22 (2012).
这篇论文探讨了使用深度信念网络进行声学建模的方法。

1.   Dahl, G. E., Yu, D., Deng, L. & Acero, A. Context-dependent pre-trained deep neural networks for large vocabulary speech recognition. IEEE Trans. Audio Speech Lang. Process. 20, 33–42 (2012).
这篇论文讨论了用于大词汇量语音识别的上下文依赖的预训练深度神经网络。

1.   Bengio, Y., Courville, A. & Vincent, P. Representation learning: a review and new perspectives. IEEE Trans. Pattern Anal. Machine Intell. 35, 1798–1828 (2013).
这篇论文回顾了表示学习，并提出了新的视角。

1.   LeCun, Y. et al. Handwritten digit recognition with a back-propagation network. In Proc. Advances in Neural Information Processing Systems 396–404 (1990).
这篇论文是关于使用反向传播网络进行手写数字识别的第一篇论文。

1.   LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. Gradient-based learning applied to document recognition. Proc. IEEE 86, 2278–2324 (1998).
这篇概述论文讨论了如何使用基于梯度的优化对模块化系统（如深度神经网络）进行端到端训练的原则，并展示了如何将神经网络（特别是卷积网络）与搜索或推理机制结合起来，以模拟复杂输出，如与文档内容相关的字符序列。

42.  Hubel, D. H. & Wiesel, T. N. Receptive ﬁelds, binocular interaction, and functional architecture in the cat’s visual cortex. J. Physiol. 160, 106–154 (1962).
这篇论文探讨了猫的视觉皮层中的感受野、双眼交互和功能架构。

1.   Felleman, D. J. & Essen, D. C. V. Distributed hierarchical processing in the primate cerebral cortex. Cereb. Cortex 1, 1–47 (1991).
这篇论文讨论了灵长类动物大脑皮层中的分布式层次处理。

1.   Cadieu, C. F. et al. Deep neural networks rival the representation of primate it cortex for core visual object recognition. PLoS Comp. Biol. 10, e1003963 (2014).
这篇论文比较了深度神经网络与灵长类动物it皮层在核心视觉对象识别方面的表示能力。

1.   Fukushima, K. & Miyake, S. Neocognitron: a new algorithm for pattern recognition tolerant of deformations and shifts in position. Pattern Recognition 15, 455–469 (1982).
这篇论文介绍了一种新的模式识别算法——新认知晶体管，它能够容忍形状变形和位置移动。

1.   Waibel, A., Hanazawa, T., Hinton, G. E., Shikano, K. & Lang, K. Phoneme recognition using time-delay neural networks. IEEE Trans. Acoustics Speech Signal Process. 37, 328–339 (1989).
这篇论文探讨了使用时延神经网络进行音素识别。

1.   Bottou, L., Fogelman-Soulié, F., Blanchet, P. & Lienard, J. Experiments with time delay networks and dynamic time warping for speaker independent isolated digit recognition. In Proc. EuroSpeech 89 537–540 (1989).
这篇论文介绍了使用时延网络和动态时间规整进行说话人独立隔离数字识别的实验。

1.   Simard, D., Steinkraus, P. Y. & Platt, J. C. Best practices for convolutional neural networks. In Proc. Document Analysis and Recognition 958–963 (2003).
这篇论文讨论了卷积神经网络的最佳实践。

1.   Vaillant, R., Monrocq, C. & LeCun, Y. Original approach for the localisation of objects in images. In Proc. Vision, Image, and Signal Processing 141, 245–250 (1994).
这篇论文提出了一种图像中对象定位的原创方法。

1.   Nowlan, S. & Platt, J. in Neural Information Processing Systems 901–908 (1995).
这篇论文讨论了神经信息处理系统中的一些主题。

1.   Lawrence, S., Giles, C. L., Tsoi, A. C. & Back, A. D. Face recognition: a convolutional neural-network approach. IEEE Trans. Neural Networks 8, 98–113 (1997).
这篇论文探讨了一种使用卷积神经网络进行面部识别的方法。

1.   Ciresan, D., Meier, U. Masci, J. & Schmidhuber, J. Multi-column deep neural network for traffic sign classification. Neural Networks 32, 333–338 (2012).
这篇论文介绍了一种用于交通标志分类的多列深度神经网络。

1.   Ning, F. et al. Toward automatic phenotyping of developing embryos from videos. IEEE Trans. Image Process. 14, 1360–1371 (2005).
这篇论文探讨了从视频中自动表型发育胚胎的方法。

1.   Turaga, S. C. et al. Convolutional networks can learn to generate affinity graphs for image segmentation. Neural Comput. 22, 511–538 (2010).
这篇论文讨论了卷积网络可以学习生成图像分割的亲和图。

1.   Garcia, C. & Delakis, M. Convolutional face ﬁnder: a neural architecture for fast and robust face detection. IEEE Trans. Pattern Anal. Machine Intell. 26, 1408–1423 (2004).
这篇论文介绍了一种用于快速和稳健面部检测的卷积面部查找器神经架构。

1.   Osadchy, M., LeCun, Y. & Miller, M. Synergistic face detection and pose estimation with energy-based models. J. Mach. Learn. Res. 8, 1197–1215 (2007).
这篇论文探讨了使用基于能量的模型进行协同面部检测和姿态估计。

1.   Tompson, J., Goroshin, R. R., Jain, A., LeCun, Y. Y. & Bregler, C. C. Efﬁcient object localization using convolutional networks. In Proc. Conference on Computer Vision and Pattern Recognition http://arxiv.org/abs/1411.4280 (2014).
这篇论文介绍了使用卷积网络进行高效对象定位的方法。

1.   Taigman, Y., Yang, M., Ranzato, M. & Wolf, L. Deepface: closing the gap to human-level performance in face verification. In Proc. Conference on Computer Vision and Pattern Recognition 1701–1708 (2014).
这篇论文讨论了Deepface技术，它在面部验证方面接近人类水平的表现。

1.   Hadsell, R. et al. Learning long-range vision for autonomous off-road driving. J. Field Robot. 26, 120–144 (2009).
这篇论文探讨了为自主越野驾驶学习远距离视觉的方法。

1.   Farabet, C., Couprie, C., Najman, L. & LeCun, Y. Scene parsing with multiscale feature learning, purity trees, and optimal covers. In Proc. International Conference on Machine Learning http://arxiv.org/abs/1202.2160 (2012).
这篇论文介绍了使用多尺度特征学习、纯度树和最优覆盖进行场景解析的方法。

1.   Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. Dropout: a simple way to prevent neural networks from overﬁtting. J. Machine Learning Res. 15, 1929–1958 (2014).
这篇论文介绍了Dropout技术，这是一种简单的防止神经网络过拟合的方法。

1.   Sermanet, P. et al. Overfeat: integrated recognition, localization and detection using convolutional networks. In Proc. International Conference on Learning Representations http://arxiv.org/abs/1312.6229 (2014).
这篇论文介绍了Overfeat，一个使用卷积网络进行集成识别、定位和检测的系统。

1.   Girshick, R., Donahue, J., Darrell, T. & Malik, J. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proc. Conference on Computer Vision and Pattern Recognition 580–587 (2014).
这篇论文讨论了为了准确对象检测和语义分割的丰富特征层次结构。

1.  Simonyan, K. & Zisserman, A. Very deep convolutional networks for large-scale image recognition. In Proc. International Conference on Learning Representations http://arxiv.org/abs/1409.1556 (2014).
这篇论文介绍了非常深的卷积网络，用于大规模图像识别。

1.  Boser, B., Sackinger, E., Bromley, J., LeCun, Y. & Jackel, L. An analog neural network processor with programmable topology. J. Solid State Circuits 26, 2017–2025 (1991).
这篇论文描述了一种具有可编程拓扑的模拟神经网络处理器。

1.  Farabet, C. et al. Large-scale FPGA-based convolutional networks. In Scaling up Machine Learning: Parallel and Distributed Approaches (eds Bekkerman, R., Bilenko, M. & Langford, J.) 399–419 (Cambridge Univ. Press, 2011).
这本书的章节讨论了基于FPGA的大规模卷积网络。

1.  Bengio, Y. Learning Deep Architectures for AI (Now, 2009).
这篇论文讨论了为人工智能学习深度架构。

1.  Montufar, G. & Morton, J. When does a mixture of products contain a product of mixtures? J. Discrete Math. 29, 321–347 (2014).
这篇数学论文探讨了混合产品何时包含混合的产品。

1.  Montufar, G. F., Pascanu, R., Cho, K. & Bengio, Y. On the number of linear regions of deep neural networks. In Proc. Advances in Neural Information Processing Systems 27 2924–2932 (2014).
这篇论文研究了深度神经网络的线性区域数量。

1.  Bengio, Y., Ducharme, R. & Vincent, P. A neural probabilistic language model. In Proc Advances in Neural Information Processing Systems 13 932–938 (2001).
这篇论文介绍了神经语言模型，它们学习将单词符号转换为单词向量或由学习到的语义特征组成的单词嵌入，以预测序列中的下一个单词。

1.  Cho, K. et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Proc. Conference on Empirical Methods in Natural Language Processing 1724–1734 (2014).
这篇论文讨论了使用RNN编码器-解码器学习短语表示，用于统计机器翻译。

1.  Schwenk, H. Continuous space language models. Computer Speech Lang. 21, 492–518 (2007).
这篇论文探讨了连续空间语言模型。

1.  Socher, R., Lin, C. C-Y., Manning, C. & Ng, A. Y. Parsing natural scenes and natural language with recursive neural networks. In Proc. International Conference on Machine Learning 129–136 (2011).
这篇论文讨论了使用递归神经网络解析自然场景和自然语言。

1.  Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. Distributed representations of words and phrases and their compositionality. In Proc. Advances in Neural Information Processing Systems 26 3111–3119 (2013).
这篇论文讨论了单词和短语的分布式表示及其组合性。

1.  Bahdanau, D., Cho, K. & Bengio, Y. Neural machine translation by jointly learning to align and translate. In Proc. International Conference on Learning Representations http://arxiv.org/abs/1409.0473 (2015).
这篇论文介绍了通过联合学习对齐和翻译进行神经机器翻译。

1.  Hochreiter, S. Untersuchungen zu dynamischen neuronalen Netzen [in German] Diploma thesis, T.U. Münich (1991).
这篇德语学位论文探讨了动态神经网络的研究。

1.  Bengio, Y., Simard, P. & Frasconi, P. Learning long-term dependencies with gradient descent is difficult. IEEE Trans. Neural Networks 5, 157–166 (1994).
这篇论文讨论了使用梯度下降学习长期依赖性是困难的。

1.  Hochreiter, S. & Schmidhuber, J. Long short-term memory. Neural Comput. 9, 1735–1780 (1997).
这篇论文介绍了LSTM循环网络，它们在最近的循环网络进展中成为了关键成分，因为它们擅长学习长期依赖性。

1.  ElHihi, S. & Bengio, Y. Hierarchical recurrent neural networks for long-term dependencies. In Proc Advances in Neural Information Processing Systems 8 http://papers.nips.cc/paper/1102-hierarchical-recurrent-neural-networks-forlong-term-dependencies (1995).
这篇论文讨论了用于长期依赖性的层次化循环神经网络。

1.  Sutskever, I. Training Recurrent Neural Networks. PhD thesis, Univ. Toronto (2012).
这篇博士学位论文讨论了训练循环神经网络。

1.  Pascanu, R., Mikolov, T. & Bengio, Y. On the difficulty of training recurrent neural networks. In Proc. 30th International Conference on Machine Learning 1310– 1318 (2013).
这篇论文讨论了训练循环神经网络的困难。

1.  Sutskever, I., Martens, J. & Hinton, G. E. Generating text with recurrent neural networks. In Proc. 28th International Conference on Machine Learning 1017– 1024 (2011).
这篇论文介绍了使用循环神经网络生成文本。

1.  Lakoff, G. & Johnson, M. Metaphors We Live By (Univ. Chicago Press, 2008).
这本书探讨了我们生活中隐喻的使用。

1.  Rogers, T. T. & McClelland, J. L. Semantic Cognition: A Parallel Distributed Processing Approach (MIT Press, 2004).
这本书讨论了语义认知的并行分布式处理方法。

1.  Xu, K. et al. Show, attend and tell: Neural image caption generation with visual attention. In Proc. International Conference on Learning Representations http:// arxiv.org/abs/1502.03044 (2015).
这篇论文介绍了一种使用视觉注意力的神经图像字幕生成方法。

1.  Graves, A., Mohamed, A.-R. & Hinton, G. Speech recognition with deep recurrent neural networks. In Proc. International Conference on Acoustics, Speech and Signal Processing 6645–6649 (2013).
这篇论文讨论了使用深度循环神经网络进行语音识别。

1.  Graves, A., Wayne, G. & Danihelka, I. Neural Turing machines. http://arxiv.org/abs/1410.5401 (2014).
这篇论文介绍了神经图灵机。

1.  Weston, J., Chopra, S. & Bordes, A. Memory networks. http://arxiv.org/abs/1410.3916 (2014).
这篇论文介绍了记忆网络。

90. Weston, J., Bordes, A., Chopra, S. & Mikolov, T. Towards AI-complete question answering: a set of prerequisite toy tasks. http://arxiv.org/abs/1502.05698 (2015).
这篇论文讨论了朝向AI完备问题回答的一系列预备玩具任务。

1.  Hinton, G. E., Dayan, P., Frey, B. J. & Neal, R. M. The wake-sleep algorithm for unsupervised neural networks. Science 268, 1558–1161 (1995).
这篇论文介绍了用于无监督神经网络的唤醒-睡眠算法。

1.  Salakhutdinov, R. & Hinton, G. Deep Boltzmann machines. In Proc. International Conference on Artificial Intelligence and Statistics 448–455 (2009).
这篇论文讨论了深度玻尔兹曼机。

1.  Vincent, P., Larochelle, H., Bengio, Y. & Manzagol, P.-A. Extracting and composing robust features with denoising autoencoders. In Proc. 25th International Conference on Machine Learning 1096–1103 (2008).
这篇论文介绍了使用去噪自编码器提取和组合鲁棒特征。

1.  Kavukcuoglu, K. et al. Learning convolutional feature hierarchies for visual recognition. In Proc. Advances in Neural Information Processing Systems 23 1090–1098 (2010).
这篇论文讨论了用于视觉识别的学习卷积特征层次结构。

1.  Gregor, K. & LeCun, Y. Learning fast approximations of sparse coding. In Proc. International Conference on Machine Learning 399–406 (2010).
这篇论文讨论了学习稀疏编码的快速近似。

1.  Ranzato, M., Mnih, V., Susskind, J. M. & Hinton, G. E. Modeling natural images using gated MRFs. IEEE Trans. Pattern Anal. Machine Intell. 35, 2206–2222 (2013).
这篇论文探讨了使用门控马尔可夫随机场（MRFs）对自然图像进行建模。

1.  Bengio, Y., Thibodeau-Laufer, E., Alain, G. & Yosinski, J. Deep generative stochastic networks trainable by backprop. In Proc. 31st International Conference on Machine Learning 226–234 (2014).
这篇论文介绍了一种可以通过反向传播训练的深度生成随机网络。

1.  Kingma, D., Rezende, D., Mohamed, S. & Welling, M. Semi-supervised learning with deep generative models. In Proc. Advances in Neural Information Processing Systems 27 3581–3589 (2014).
这篇论文讨论了使用深度生成模型进行半监督学习。

1.  Ba, J., Mnih, V. & Kavukcuoglu, K. Multiple object recognition with visual attention. In Proc. International Conference on Learning Representations http:// arxiv.org/abs/1412.7755 (2014).
这篇论文介绍了使用视觉注意力进行多对象识别的方法。

1.   Mnih, V. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).
这篇论文探讨了通过深度强化学习实现人类水平的控制。

1.   Bottou, L. From machine learning to machine reasoning. Mach. Learn. 94, 133–149 (2014).
这篇文章讨论了从机器学习到机器推理的发展。

1.   Vinyals, O., Toshev, A., Bengio, S. & Erhan, D. Show and tell: a neural image caption generator. In Proc. International Conference on Machine Learning http:// arxiv.org/abs/1502.03044 (2014).
这篇论文介绍了一种神经图像字幕生成器，能够根据图像生成描述性文本。

1.   van der Maaten, L. & Hinton, G. E. Visualizing data using t-SNE. J. Mach. Learn. Research 9, 2579–2605 (2008).
这篇论文介绍了使用t-SNE算法进行数据可视化的方法。

作者信息 转载和权限信息可在 www.nature.com/reprints 找到。作者声明没有竞争性的财务利益。读者可以在 go.nature.com/7cjbaa 上对本文的在线版本发表评论。通信应发送至Y.L. (yann@cs.nyu.edu)。

这是对文档中列出的参考文献部分的翻译摘要。这些参考文献涵盖了深度学习领域的多个关键研究和发现，包括卷积神经网络、循环神经网络、无监督学习、强化学习以及深度学习在图像识别、语音识别和自然语言处理等方面的应用。每一项研究都在推动人工智能和机器学习的进步，为未来的技术发展奠定了基础。