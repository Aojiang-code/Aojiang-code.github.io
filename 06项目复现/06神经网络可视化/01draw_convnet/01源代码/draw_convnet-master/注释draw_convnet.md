"""
版权所有 (c) 2017, 丁伟光 (Gavin Weiguang Ding)
保留所有权利。

在满足以下条件的情况下，允许重新分发和使用源代码和二进制形式，无论是否修改：

1. 重新分发源代码时，必须保留上述版权声明、此条件列表以及随附的免责声明。

2. 以二进制形式重新分发时，必须在随分发提供的文档和/或其他材料中复制上述版权声明、此条件列表以及随附的免责声明。

3. 未经特定事先书面许可，不得使用版权持有者或其贡献者的名字来支持或推广从本软件衍生出的产品。

本软件由版权持有者和贡献者“按原样”提供，并且任何明示或暗示的保证，包括但不限于对适销性和特定用途适用性的暗示保证，均被排除。在任何情况下，即使被告知可能发生此类损害的可能性，版权持有者或贡献者也不应对任何直接的、间接的、偶然的、特殊的、惩罚性的或后果性的损害（包括但不限于替代商品或服务的采购；使用、数据或利润的损失；或业务中断）承担责任，无论是基于合同责任、严格责任还是侵权行为（包括疏忽或其他原因）从使用本软件中以任何方式产生。
"""


## 导入库
```python
import os  # 导入Python的标准库os模块，用于与操作系统交互，例如文件路径操作。
import numpy as np  # 导入NumPy库并简称为np，NumPy是一个强大的科学计算库，特别擅长处理大型多维数组和矩阵。
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块并简称为plt，用于创建静态、交互式和动画可视化。
plt.rcdefaults()  # 重置matplotlib的配置为默认设置，这通常用于确保绘图样式在不同环境中保持一致。
from matplotlib.lines import Line2D  # 从matplotlib的lines模块中导入Line2D类，用于创建和定制线条对象。
from matplotlib.patches import Rectangle  # 从matplotlib的patches模块中导入Rectangle类，用于创建和定制矩形图形。
from matplotlib.patches import Circle  # 从matplotlib的patches模块中导入Circle类，用于创建和定制圆形图形。
```

这段代码主要是导入了一些Python编程中常用的库，用于数据处理、数学计算、以及绘图。导入这些库后，你可以使用它们提供的功能来执行各种任务，比如数学运算、数据可视化等。

## 定义不同的颜色和层的数量

```python
# 定义一个常量NumDots，值为4，这可能表示在绘图中使用的点的数量。
NumDots = 4

# 定义一个常量NumConvMax，值为8，这可能表示卷积层中的最大特征图数量。
NumConvMax = 8#表示，最多8张特征图

# 定义一个常量NumFcMax，值为20，这可能表示全连接层中的最大单元数量。
NumFcMax = 20#表示，全连接层中有20个方块展示出来

# 定义一个常量White，值为1.0，这可能表示绘图中白色的颜色值。
White = 1.

# 定义一个常量Light，值为0.7，这可能表示绘图中浅灰色的颜色值。
Light = 0.7

# 定义一个常量Medium，值为0.5，这可能表示绘图中中等灰色的颜色值。
Medium = 0.5

# 定义一个常量Dark，值为0.3，这可能表示绘图中深灰色的颜色值。
Dark = 0.3

# 定义一个常量Darker，值为0.15，这可能表示绘图中更深的灰色的颜色值。
Darker = 0.15

# 定义一个常量Black，值为0.0，这可能表示绘图中黑色的颜色值。
Black = 0.
```

这些常量看起来是用于一个绘图脚本中，用于定义不同的颜色和层的数量，以便在可视化神经网络结构时使用。



## 定义一个名为 `add_layer` 的函数，在给定的图形上添加一层矩形元素，并且为这些元素指定颜色

这段代码定义了一个名为 `add_layer` 的函数，它的作用是在给定的图形上添加一层矩形元素，并且为这些元素指定颜色。下面是对这段代码的逐行中文注释：

```python
# 定义一个函数add_layer，它负责在图形上添加一层矩形元素。
def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              ):
    # "add a rectangle" 是对接下来操作的英文注释，说明将要添加一个矩形。
    # 将传入的top_left参数转换为NumPy数组，以便于进行数学运算。
    top_left = np.array(top_left)
    # 将传入的loc_diff参数转换为NumPy数组。
    loc_diff = np.array(loc_diff)
    # 计算第一个矩形的起始位置，它是top_left向下偏移一个size[0]的值。
    loc_start = top_left - np.array([0, size[0]])
    # 循环num次，每次创建一个矩形并添加到patches列表中。
    for ind in range(num):
        # 对于每次循环，计算当前矩形的位置，并创建一个Rectangle对象。
        # 这个Rectangle对象的宽度是size[1]，高度是size[0]。
        # 矩形的位置是通过将loc_start与循环索引ind乘以loc_diff来确定的。
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        # 根据循环索引ind是否为偶数，决定矩形的颜色是Medium还是Light，并添加到colors列表中。
        # 如果ind是偶数，则颜色为Light，否则为Medium。
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)
```

这个函数可能用于神经网络结构图的可视化，其中`patches`是一个用于存储图形元素的列表，`colors`是存储对应颜色的列表。`size`参数定义了每个矩形的大小，`num`参数定义了要创建的矩形数量，`top_left`定义了第一个矩形的起始位置，`loc_diff`定义了连续矩形之间的间距。

## 定义一个名为 `add_layer_with_omission` 的函数，作用是在图形上添加一层矩形元素，并且可以根据参数选择性地省略一些元素

这段代码定义了一个名为 `add_layer_with_omission` 的函数，它的作用是在图形上添加一层矩形元素，并且可以根据参数选择性地省略一些元素。下面是对这段代码的逐行中文注释：

```python
# 定义一个函数add_layer_with_omission，它负责在图形上添加一层矩形元素，并且可以根据参数省略一些元素。
def add_layer_with_omission(patches, colors, size=(24, 24),
                            num=5, num_max=8,
                            num_dots=4,
                            top_left=[0, 0],
                            loc_diff=[3, -3],
                            ):
    # 将传入的top_left参数转换为NumPy数组，以便于进行数学运算。
    top_left = np.array(top_left)
    # 将传入的loc_diff参数转换为NumPy数组。
    loc_diff = np.array(loc_diff)
    # 计算第一个矩形的起始位置，它是top_left向下偏移一个size[0]的值。
    loc_start = top_left - np.array([0, size[0]])
    # 确定实际绘制的矩形数量，取num和num_max中的较小值。
    this_num = min(num, num_max)
    # 计算开始省略的索引位置，用于确定从哪个矩形开始绘制省略标记。
    start_omit = (this_num - num_dots) // 2
    # 计算结束省略的索引位置。
    end_omit = this_num - start_omit
    # 对start_omit进行调整，使其成为实际的索引值。
    start_omit -= 1
    # 循环遍历this_num次，每次创建一个图形元素（矩形或圆）并添加到patches列表中。
    for ind in range(this_num):
        # 判断当前索引ind是否处于需要省略的范围内，如果num大于num_max，并且ind在start_omit和end_omit之间，则设置omit为True。
        if (num > num_max) and (start_omit < ind < end_omit):
            omit = True
        else:
            # 如果当前索引不在省略范围内，则设置omit为False。
            omit = False

        # 如果omit为True，则在patches列表中添加一个半径为0.5的Circle对象，代表省略的元素。
        if omit:
            patches.append(
                Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.5))
        else:
            # 如果omit为False，则添加一个Rectangle对象，表示正常的层元素。
            patches.append(Rectangle(loc_start + ind * loc_diff,
                                     size[1], size[0]))

        # 根据omit的值和当前索引ind，确定并添加相应的颜色到colors列表中。
        if omit:
            # 如果省略，则颜色为Black。
            colors.append(Black)
        elif ind % 2:
            # 如果当前索引为奇数，则颜色为Medium。
            colors.append(Medium)
        else:
            # 如果当前索引为偶数，则颜色为Light。
            colors.append(Light)
```

这个函数可能用于神经网络结构图的可视化，其中`patches`是一个用于存储图形元素的列表，`colors`是存储对应颜色的列表。`size`参数定义了每个元素的大小，`num`参数定义了要创建的元素数量，`num_max`定义了最大显示数量，`num_dots`定义了省略标记的数量，`top_left`定义了第一个元素的起始位置，`loc_diff`定义了连续元素之间的间距。当`num`大于`num_max`时，会在中间的某些元素位置用圆圈标记表示省略。

## 定义一个名为 `add_mapping` 的函数，作用是在图形上添加表示层与层之间映射关系的图形元素，如矩形和线条

这段代码定义了一个名为 `add_mapping` 的函数，它的作用是在图形上添加表示层与层之间映射关系的图形元素，如矩形和线条。下面是对这段代码的逐行中文注释：

```python
# 定义一个函数add_mapping，用于在图形上添加表示层之间映射关系的元素。
def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):
    # 根据当前层的索引ind_bgn，计算起始映射点的位置。
    # 起始映射点的位置由当前层的左上角位置、层内元素数量和元素间距决定，
    # 并根据start_ratio对起始映射点进行偏移。
    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([
            start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),  # 水平方向的偏移量
            - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])  # 垂直方向的偏移量
        ])

    # 根据下一层的索引(ind_bgn + 1)，计算结束映射点的位置。
    # 结束映射点的位置由下一层的左上角位置、层内元素数量和元素间距决定，
    # 并根据end_ratio对结束映射点进行偏移。
    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) * np.array(loc_diff_list[ind_bgn + 1]) \
        + np.array([
            end_ratio[0] * size_list[ind_bgn + 1][1],  # 水平方向的偏移量
            - end_ratio[1] * size_list[ind_bgn + 1][0]  # 垂直方向的偏移量
        ])

    # 在patches列表中添加一个Rectangle对象，表示从start_loc到end_loc的映射区域。
    # patch_size[1]是映射区域的宽度，-patch_size[0]是映射区域的高度（负值表示向上）。
    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
    # 为映射区域添加颜色Dark。
    colors.append(Dark)

    # 在patches列表中添加一条Line2D对象，表示从start_loc到end_loc的直线映射。
    patches.append(Line2D([start_loc[0], end_loc[0]], [start_loc[1], end_loc[1]]))
    # 为直线映射添加颜色Darker。
    colors.append(Darker)

    # 在patches列表中添加另一条Line2D对象，表示从start_loc的右侧到end_loc的直线映射。
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]], [start_loc[1], end_loc[1]]))
    # 为这条直线映射添加颜色Darker。
    colors.append(Darker)

    # 在patches列表中添加第三条Line2D对象，表示从start_loc底部到end_loc的直线映射。
    patches.append(Line2D([start_loc[0], end_loc[0]], [start_loc[1] - patch_size[0], end_loc[1]]))
    # 为这条直线映射添加颜色Darker。
    colors.append(Darker)

    # 在patches列表中添加第四条Line2D对象，表示从start_loc底部右侧到end_loc的直线映射。
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]], [start_loc[1] - patch_size[0], end_loc[1]]))
    # 为这条直线映射添加颜色Darker。
    colors.append(Darker)
```

这段代码中，`patches` 是一个用于存储图形元素（如矩形和线条）的列表，`colors` 是一个用于存储这些图形元素颜色的列表。`start_ratio` 和 `end_ratio` 定义了起始和结束映射点相对于层大小的比例偏移。`patch_size` 定义了映射区域的大小。`ind_bgn` 是当前层的索引，`top_left_list`、`loc_diff_list`、`num_show_list` 和 `size_list` 分别存储了每层的左上角位置、层与层之间的偏移、每层显示的数量以及每层的大小。通过计算 `start_loc` 和 `end_loc`，函数在图形上绘制了表示层之间映射的矩形和线条。

## 定义一个用于在图形上添加文本标签的函数 `label`，并在主程序部分设置了一些绘图相关的变量，以及初始化了绘图环境
这段代码定义了一个用于在图形上添加文本标签的函数 `label`，并在主程序部分设置了一些绘图相关的变量，以及初始化了绘图环境。下面是逐行中文注释：

```python
# 定义一个函数label，用于在matplotlib图形上添加文本标签。
def label(xy, text, xy_off=[0, 4]):
    # 在点xy的位置添加文本标签，文本内容为text。
    # xy_off是文本标签相对于xy位置的偏移量，xy_off默认为[0, 4]，意味着文本标签默认向右偏移4个单位。
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             # 设置文本字体为无衬线字体'sans-serif'，大小为8。
             family='sans-serif', size=8)

# 判断是否是直接运行该脚本，而不是作为模块导入到其他脚本中。
if __name__ == '__main__':
    # 设置全连接层单元的大小为2x2。
    fc_unit_size = 2
    # 设置层与层之间的水平间隔为40个单位。
    layer_width = 40
    # 设置一个标志变量flag_omit，用于控制是否在绘图时省略某些层的显示。
    # 如果flag_omit为True，则在绘图时会省略显示某些层。
    flag_omit = True

    # 初始化patches列表，用于存储matplotlib的图形元素，如矩形、圆形等。
    patches = []
    # 初始化colors列表，用于存储与patches中图形元素对应的颜色。
    colors = []

    # 使用matplotlib的pyplot接口创建一个图形(fig)和一个坐标轴(ax)。
    fig, ax = plt.subplots()
    # 这一行代码会创建一个图形窗口，并准备好绘图的环境。
```

这段代码是绘图脚本的一部分，其中`label`函数用于在指定位置添加文本标签，`patches`和`colors`列表用于存储绘图元素和颜色，`fig`和`ax`是matplotlib中用于绘图的主要对象。`if __name__ == '__main__':`这一行确保了当这个脚本被直接运行时，下面的代码块会被执行。


   
   
   
   
   
   
   
   
## 卷积层部分
   
    ############################
这段代码是用于设置和绘制卷积神经网络中的卷积层部分的Python脚本。以下是逐行中文注释：

```python
################################
# conv layers
# 定义每个卷积层的特征图尺寸列表。
size_list = [(32, 32), (18, 18), (10, 10), (6, 6), (4, 4)]

# 定义每个卷积层的特征图数量列表。
num_list = [3, 32, 32, 48, 48]

# 定义卷积层之间的水平间距列表。
x_diff_list = [0, layer_width, layer_width, layer_width, layer_width]

# 定义每个卷积层的文本标签列表，第一个元素为'Inputs'，其余为'Feature\nmaps'。
text_list = ['Inputs'] + ['Feature\nmaps'] * (len(size_list) - 1)

# 定义卷积层内特征图之间的间距，所有层使用相同的间距。
loc_diff_list = [[3, -3]] * len(size_list)

# 根据NumConvMax限制num_list中的数值，确保不超过NumConvMax，创建num_show_list。
num_show_list = list(map(min, num_list, [NumConvMax] * len(num_list)))

# 计算每个卷积层左上角的起始位置，创建top_left_list。
top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

# 反向遍历size_list，用于绘制卷积层。
for ind in range(len(size_list)-1,-1,-1):
    # 如果flag_omit为True，则调用add_layer_with_omission函数绘制带有省略的层。
    if flag_omit:
        add_layer_with_omission(patches, colors, size=size_list[ind],
                                num=num_list[ind],
                                num_max=NumConvMax,
                                num_dots=NumDots,
                                top_left=top_left_list[ind],
                                loc_diff=loc_diff_list[ind])
    # 否则，调用add_layer函数绘制正常的层。
    else:
        add_layer(patches, colors, size=size_list[ind],
                  num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
    # 使用label函数在每个卷积层的左上角位置绘制文本标签，包含特征图数量和尺寸。
    label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
        num_list[ind], size_list[ind][0], size_list[ind][1]))
```

这段代码中，`size_list` 存储了每个卷积层输出的特征图尺寸，`num_list` 存储了每个卷积层的特征图数量。`x_diff_list` 定义了每层之间的水平间隔，`text_list` 定义了每层的文本标签，`loc_diff_list` 定义了每层内部特征图之间的间距。`num_show_list` 根据 `NumConvMax` 限制 `num_list` 中的数值，`top_left_list` 计算每层的起始位置。最后，通过 `for` 循环反向遍历每层，并根据 `flag_omit` 的值决定是使用 `add_layer_with_omission` 函数还是 `add_layer` 函数来绘制层，最后使用 `label` 函数在每层绘制文本标签。


## 层间连接

这段代码是用于设置和绘制卷积神经网络中的层间连接（例如卷积和池化层）部分的Python脚本。以下是逐行中文注释：

```python
################################
# in between layers
# 定义层间连接的起始偏移比例列表。
start_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]

# 定义层间连接的结束偏移比例列表。
end_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]

# 定义层间连接的映射区域大小列表。
patch_size_list = [(5, 5), (2, 2), (5, 5), (2, 2)]

# 创建一个索引列表，用于遍历层间连接。
ind_bgn_list = range(len(patch_size_list))

# 定义层间连接的文本标签列表，包含'Convolution'和'Max-pooling'。
text_list = ['Convolution', 'Max-pooling', 'Convolution', 'Max-pooling']

# 遍历每一层间连接。
for ind in range(len(patch_size_list)):
    # 调用add_mapping函数来添加层间连接的图形元素，包括矩形和线条。
    # 这里传入了起始偏移比例、结束偏移比例、映射区域大小、当前连接的索引，
    # 以及之前定义的top_left_list、loc_diff_list、num_show_list和size_list。
    add_mapping(
        patches, colors, start_ratio_list[ind], end_ratio_list[ind],
        patch_size_list[ind], ind,
        top_left_list, loc_diff_list, num_show_list, size_list
    )
    # 使用label函数在适当的位置添加文本标签，描述当前层间连接的类型和卷积核大小。
    # xy_off参数用于调整文本标签的位置，使其更加美观。
    label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel'.format(
        patch_size_list[ind][0], patch_size_list[ind][1]), xy_off=[26, -65]
    )
```

这段代码中，`start_ratio_list` 和 `end_ratio_list` 定义了层间连接的起始和结束偏移比例，`patch_size_list` 定义了映射区域的大小。`ind_bgn_list` 是一个索引列表，用于遍历每一层间连接。`text_list` 包含了每一层间连接的文本标签，如 "Convolution"（卷积）和 "Max-pooling"（最大池化）。通过 `for` 循环，代码调用 `add_mapping` 函数来绘制层间连接，并使用 `label` 函数在适当的位置添加描述性文本标签。`xy_off` 参数用于调整文本标签的位置，确保标签不会与图形元素重叠，提高可视化的可读性。


## 全连接层部分

这段代码是用于设置和绘制卷积神经网络中的全连接层部分的Python脚本。以下是逐行中文注释：

```python
################################
# fully connected layers
# 定义全连接层的尺寸列表，每个全连接层的大小为fc_unit_size x fc_unit_size。
size_list = [(fc_unit_size, fc_unit_size)] * 3

# 定义全连接层的单元数量列表。
num_list = [768, 500, 2]

# 根据NumFcMax限制num_list中的数值，创建num_show_list。
num_show_list = list(map(min, num_list, [NumFcMax] * len(num_list)))

# 定义全连接层之间的水平间距列表，基于之前的x_diff_list和layer_width。
x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width]

# 计算每个全连接层左上角的起始位置，创建top_left_list。
top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

# 定义全连接层内单元之间的间距，所有层使用相同的间距。
loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)

# 定义全连接层的文本标签列表，包含'Hidden units'和'Outputs'。
text_list = ['Hidden\nunits'] * (len(size_list) - 1) + ['Outputs']

# 遍历每个全连接层。
for ind in range(len(size_list)):
    # 如果flag_omit为True，则调用add_layer_with_omission函数绘制带有省略的层。
    if flag_omit:
        add_layer_with_omission(patches, colors, size=size_list[ind],
                                num=num_list[ind],
                                num_max=NumFcMax,
                                num_dots=NumDots,
                                top_left=top_left_list[ind],
                                loc_diff=loc_diff_list[ind])
    # 否则，调用add_layer函数绘制正常的层。
    else:
        add_layer(patches, colors, size=size_list[ind],
                  num=num_show_list[ind],
                  top_left=top_left_list[ind],
                  loc_diff=loc_diff_list[ind])
    # 使用label函数在每个全连接层的左上角位置绘制文本标签，包含单元数量。
    label(top_left_list[ind], text_list[ind] + '\n{}'.format(
        num_list[ind]))

# 更新text_list，用于接下来的标签绘制，包含'Flatten', 'Fully connected', 'Fully connected'。
text_list = ['Flatten\n', 'Fully\nconnected', 'Fully\nconnected']

# 再次遍历每个全连接层，这次用于绘制层的描述性标签。
for ind in range(len(size_list)):
    # 使用label函数在每个全连接层的位置绘制描述性文本标签。
    # xy_off参数用于调整文本标签的位置，使其更加美观。
    label(top_left_list[ind], text_list[ind], xy_off=[-10, -65])
```

这段代码中，`size_list` 存储了全连接层的尺寸，`num_list` 存储了每个全连接层的单元数量。`num_show_list` 根据 `NumFcMax` 限制 `num_list` 中的数值。`x_diff_list` 定义了全连接层之间的水平间隔，`top_left_list` 计算每层的起始位置。`loc_diff_list` 定义了全连接层内部单元之间的间距。通过两个 `for` 循环，第一个用于绘制层并添加数量标签，第二个用于添加描述性标签，如 "Flatten"（展平层）、"Fully connected"（全连接层）。`xy_off` 参数用于调整文本标签的位置，确保标签的可读性。

## 保存为图片文件

这段代码是用于完成神经网络结构图的绘制，并将其保存为图片文件的Python脚本。以下是逐行中文注释：

```python
################################
# 遍历patches和colors列表中的每个图形元素和颜色。
for patch, color in zip(patches, colors):
    # 将颜色值乘以np.ones(3)，以确保颜色是一个长度为3的数组，对应RGB三个通道。
    patch.set_color(color * np.ones(3))
    # 检查当前的图形元素是否为Line2D类型。
    if isinstance(patch, Line2D):
        # 如果是Line2D类型，则将其添加到坐标轴ax中。
        ax.add_line(patch)
    else:
        # 如果不是Line2D类型，则设置图形元素的边缘颜色，并将其添加到坐标轴ax中。
        patch.set_edgecolor(Black * np.ones(3))
        ax.add_patch(patch)

# 调用tight_layout函数，自动调整子图参数，使之填充整个图像区域。
plt.tight_layout()

# 设置坐标轴的比例为相等，即一个单位的长度在x轴和y轴上显示的大小相同。
plt.axis('equal')

# 关闭坐标轴的显示。
plt.axis('off')

# 显示图形。
plt.show()

# 设置图像的大小为8英寸宽，2.5英寸高。
fig.set_size_inches(8, 2.5)

# 设置图像文件的保存目录。
fig_dir = './'
# 设置图像文件的扩展名。
fig_ext = '.png'
# 将图像保存为文件，bbox_inches='tight'确保所有内容都被包含在内，pad_inches=0设置无边界填充。
fig.savefig(os.path.join(fig_dir, 'convnet_fig' + fig_ext),
            bbox_inches='tight', pad_inches=0)
```

这段代码中，首先遍历 `patches` 和 `colors` 列表，为每个图形元素设置颜色，并根据元素类型（线条或图形）进行相应的处理，然后将其添加到绘图中。接着，使用 `tight_layout` 调整布局，使用 `axis` 函数关闭坐标轴显示，并使用 `show` 显示图形。之后，设置图形的大小，并将其保存为PNG格式的图片文件。`bbox_inches='tight'` 参数确保所有的图形元素都被包含在内，没有被裁切掉。`pad_inches=0` 参数表示在保存时不增加额外的边界空间。



