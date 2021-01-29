import numpy as np


def get_loc(num, _arr):  # 返回给定节点的坐标点
    _arr = np.array(_arr).reshape(3, 3)
    for i in range(len(_arr)):
        for j in range(len(_arr[i])):
            if num == _arr[i][j]:  # 找到值所在的位置并返回
                return i, j

# 定义启发函数(1.以不在位的数量值为启发函数进行度量,2.初始状态和最终状态的节点曼哈顿距离,3.宽度优先启发为0)


def val(arr, arr_final, method=0):
    if method == 1:  # 曼哈顿距离
        _arr = np.array(arr).reshape(3, 3)
        _arr_final = np.array(arr_final).reshape(3, 3)
        total = 0
        for i in range(len(_arr)):
            for j in range(len(_arr[i])):
                m, n = get_loc(_arr[i][j], arr_final)  # 找到给定值的横坐标和纵坐标
                total += np.abs(i - m) + np.abs(j - n)  # 计算所有节点的曼哈顿距离之和
        return total
    if method == 2:  # 宽度优先
        return 0
# 不在位数量
    total = []
    for i in range(len(arr)):  # 计算list中对不上的数量,0除外
        if arr[i] != 0:
            total.append(arr[i] - arr_final[i])
    return len(total) - total.count(0)


# 定义一个函数用来执行,矩阵的变换工作,即移动"0"


def arr_swap(arr, destnation, flag=False):
    if flag:  # 如果flag为true那么直接修改矩阵
        z_pos = arr.argmin()  # 获得0的位置
        tmp = arr[destnation]  # 和要修改的位置进行交换
        arr[destnation] = arr[z_pos]
        arr[z_pos] = tmp
        return list(arr)  # 返回结果
# 如果flag为false,不修改传入的矩阵,而返回一个修改后的副本
    _arr = np.array(arr.copy())  # 创建副本
    z_pos = _arr.argmin()  # 获得0的位置
    tmp = _arr[destnation]  # 交换位置
    _arr[destnation] = _arr[z_pos]
    _arr[z_pos] = tmp
    return list(_arr)  # 返回副本

# 定义一个节点类,用来储存一些相关的信息
# 节点的数据有,父节点,该状态下节点的启发值,该状态节点的矩阵,扩展到第几步


class node:
    par = None
    value = -1
    arr = None
    step = 0

    def __init__(self, p, val, a, s):  # 根据传入的值初始化节点
        self.par = p
        self.step = s
        self.value = val
        self.arr = np.array(a)

    def up(self, ss):  # 定义节点中"0"向上为一个函数,
        if np.array(self.arr).argmin() - 3 >= 0:  # 当能够向上移动时，返回向上移动后的节点
            # 返回后的节点，计算启发值，更新数组，并将新节点的值父节点设置为调用函数的节点
            tmp = np.array(self.arr).argmin() - 3
            ar = arr_swap(self.arr, tmp)
            v = val(ar, arr_final, ss)
            v += self.step + 1
            new_node = node(p=self, val=v, a=ar, s=self.step + 1)
            return new_node  # 返回生成的子节点
        else:
            return None

    def down(self, ss):
        # 定义向上的函数（“0“向上）
        # 当能够向上移动时，返回向上移动生成的子节点。
        if (np.array(self.arr).argmin() + 3) <= 8:  # 判断是否满足向上的条件
            tmp = np.array(self.arr).argmin() + 3  # 生成子节点
            ar = arr_swap(self.arr, tmp)
            v = val(ar, arr_final, ss)
            v += self.step + 1
            new_node = node(p=self, val=v, a=ar, s=self.step + 1)
            return new_node  # 返回子节点
        else:
            return None  # 不满足条件则返回空

    def left(self, ss):  # 定义向左移动函数
        if np.array(self.arr).argmin(
        ) - 1 >= 0 and np.array(self.arr).argmin() % 3 != 0:  # 如果满足可以向左移动的条件
            tmp = np.array(self.arr).argmin() - 1  # 返回向左移动后生成的子节点
            ar = arr_swap(self.arr, tmp)
            v = val(ar, arr_final, ss)
            v += self.step + 1
            new_node = node(p=self, val=v, a=ar, s=self.step + 1)
            return new_node  # 返回新的子节点
        else:
            return None  # 不满足条件返回为空

    def right(self, ss):  # 定义向右移动的函数
        if np.array(self.arr).argmin() + \
                1 <= 8 and np.array(self.arr).argmin() % 3 != 2:  # 若满足条件
            tmp = np.array(self.arr).argmin() + 1  # 返回生成的子节点

            ar = arr_swap(self.arr, tmp)
            v = val(ar, arr_final, ss)
            v += self.step + 1
            new_node = node(p=self, val=v, a=ar, s=self.step + 1)
            return new_node  # 返回子节点
        else:
            return None  # 不满足返回none


# 定义函数用来判断，一个节点是否在生成的表中


def in_open(t, openl):
    for i in openl:
        if all(i.arr == t.arr):  # 如果找到了arr相同的节点
            if t.value < i.value:  # 更新启发值为较小的一个节点
                i.value = t.value
          #      print("update")
            return True  # 该节点在open表中，返回true
    return False  # 节点不在open表中，返回false

# 定义函数用来判断，一个节点是否在已经结束生成的表中


def in_close(t, closel, openl):
    for i in closel:
        if all(i.arr == t.arr):  # 如果在结束的表中找到了相同的节点信息
            if t.value < i.value:  # 如果传入的节点启发值更优于close中的节点，那么说明有其他生成该节点的方式更优
                i.value = t.value
                openl.append(t)  # 加入该节点到open表中
#
            return True  # 在close表中，返回true
    return False  # 不在则返回false


# 开始进行循环查找
def Astar(arr_start, arr_final, val_method=1):
    start = node(None, val(arr_start, arr_final, val_method), arr_start, 0)
    # 定义两个表，open，和close用于记录正在准备生成的节点，和已经生成的节点
    open_l = []
    close_l = []
    # 将start节点加入到open表中
    open_l.append(start)
    n = start
    step = 0
    while (1):
        step += 1
        # 节点开始进行生成，向上，向下，向左，向右运行，查看满足生成条件。
        # 节点的生成分为三种情况：
        # （1）在open表中，那么将节点的启发值更新为比较优的值
        # （2）在close表中，如果新生成的节点启发值更优，则加入该节点但open表中
        # （3）如果都不在表中，那么直接加入当open表中
        #
        # len1=len(open_l)
        # cnt=0
        if n.up(val_method) is not None:
            tmp = n.up(val_method)
            f1 = in_open(tmp, open_l)  # 是否在open中
            f2 = in_close(tmp, close_l, open_l)  # 是否在close中
            if f1 == False and f2 == False:  # 两者都不在，加入open
                open_l.append(n.up(val_method))

        if n.down(val_method) is not None:
            tmp = n.down(val_method)
            f1 = in_open(tmp, open_l)   # 是否在open中
            f2 = in_close(tmp, close_l, open_l)  # 是否在close中
            if f1 == False and f2 == False:  # 两者都不在，加入open
                open_l.append(n.down(val_method))

        if n.left(val_method) is not None:
            tmp = n.left(val_method)
            f1 = in_open(tmp, open_l)   # 是否在open中
            f2 = in_close(tmp, close_l, open_l)  # 是否在close中
            if f1 == False and f2 == False:  # 两者都不在，加入open
                open_l.append(n.left(val_method))

        if n.right(val_method) is not None:
            tmp = n.right(val_method)
            f1 = in_open(tmp, open_l)  # 是否在open中
            f2 = in_close(tmp, close_l, open_l)  # 是否在close中
            if f1 == False and f2 == False:  # 两者都不在，加入open
                open_l.append(n.right(val_method))

        # 生成结束后，将生成完毕的节点移出open表中
        open_l.remove(n)
        # print("({name1},{name2},cnt={name3})".format(name1=len1,name2=len(open_l),name3=cnt))
        if len(open_l) == 0:  # 如果表元素为空，则退出
            break
        close_l.append(n)  # 将该节点加入到close表中

        node_v = []  # 新的open表中，各个节点的启发值记录
        for i in open_l:
            node_v.append(i.value)  # 加入每个节点的启发值

        min_posi = np.array(node_v).argmin()  # 找到最小的
        n = open_l[min_posi]  # 将最小的节点作为下一个循环要生成的节点
        if list(n.arr) == list(arr_final):  # 如果要生成的节点满足最终状态的需要，那么退出寻找
            break

    return start, n, step, len(close_l), len(open_l) + len(close_l)
# 定义输出


def print_put(n, start):
    final = []  # 最终的节点路线
    ptr = n  # 用于循环查找，n为最后一次生成的节点
    index = []  # 查看节点的step
    while ptr.par is not None:
        # 查找父节点。加入当final中。
        final.append(ptr.arr)
        index.append(ptr.step)
        ptr = ptr.par

    # 输出如何得到最后的输出。
    print(start.arr.reshape(3, 3))
    for i in range(len(final)):
        print("step: ", i + 1)
        print(np.array(final[len(final) - i - 1]).reshape(3, 3))


def getStatus(arr):  # 用序偶奇偶性判断是否有解,序偶相同的是一个等价集可以通过变换得到
    sum = 0
    for i in range(len(arr)):
        for j in range(0, i):
            if arr[j] < arr[i] and arr[j] != 0:
                sum += 1

    return sum % 2


if __name__ == '__main__':
    # 定义矩阵的初始状态
    # 2, 1, 3, 8, 0, 4, 7, 6, 5 一个无解的序列
    arr_start = np.array([2, 8, 3, 1, 6, 4, 7, 0, 5])
    # 定义矩阵的最终状态
    arr_final = np.array([1, 2, 3, 8, 0, 4, 7, 6, 5])
    # 判断是否有解
    if getStatus(arr_final) != getStatus(arr_start):
        print("该初始状态无解")
        exit(0)

# 启发函数为不在位数量
    print("估价函数一:")
    start, n, s, num_expand, num_generate = Astar(arr_start, arr_final, 0)
    print("循环次数:{num3}  生成节点数量:{num1},  扩展节点数量:{num2}".format(
        num1=num_generate, num2=num_expand, num3=s))
    print_put(n, start)

# 启发函数为曼哈顿距离
    print("估价函数二:")

    start, n, s, num_expand, num_generate = Astar(arr_start, arr_final, 1)
    print("循环次数:{num3}  生成节点数量:{num1},  扩展节点数量:{num2}".format(
        num1=num_generate, num2=num_expand, num3=s))
    print_put(n, start)
# 宽度优先,启发函数为0
    print("宽度优先:")

    start, n, s, num_expand, num_generate = Astar(arr_start, arr_final, 2)
    print("循环次数:{num3}  生成节点数量:{num1},  扩展节点数量:{num2}".format(
        num1=num_generate, num2=num_expand, num3=s))
    print_put(n, start)
