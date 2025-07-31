import numpy
import numpy as np
matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
target = 3


class Solution:
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        flag = False
        upIndex = 0
        downIndex = len(matrix) - 1
        midIndexRaw = upIndex + (downIndex - upIndex) // 2
        while (downIndex != (upIndex + 1)) & (downIndex != upIndex):
            if target == matrix[midIndexRaw][0]:
                flag = True
                return flag
            elif target != matrix[midIndexRaw][0]:
                if target < matrix[midIndexRaw][0]:
                    downIndex = midIndexRaw
                    midIndexRaw = upIndex + (downIndex - upIndex) // 2
                elif target > matrix[midIndexRaw][0]:
                    upIndex = midIndexRaw
                    midIndexRaw = upIndex + (downIndex - upIndex) // 2
        if (target == matrix[upIndex][0]) | (target == matrix[downIndex][0]):
            flag = True
            return flag
        elif (target < matrix[upIndex][0]):
            flag = False
            return flag
        elif (target > matrix[upIndex][0]):
            if target > matrix[downIndex][0]:
                startIndex = downIndex
            else:
                startIndex = upIndex
            # 尝试封装一下二分查找函数
            leftIndex = 0
            rightIndex = len(matrix[startIndex]) - 1
            midIndexCol = leftIndex + (rightIndex - leftIndex) // 2
            while (rightIndex != (leftIndex + 1)) & (rightIndex != leftIndex):
                if target == matrix[startIndex][midIndexCol]:
                    flag = True
                    return flag
                elif target != matrix[startIndex][midIndexCol]:
                    if target < matrix[startIndex][midIndexCol]:
                        rightIndex = midIndexCol
                        midIndexCol = leftIndex + (rightIndex - leftIndex) // 2
                    elif target > matrix[startIndex][midIndexCol]:
                        leftIndex = midIndexCol
                        midIndexCol = leftIndex + (rightIndex - leftIndex) // 2
            if (target == matrix[startIndex][leftIndex]) | (target == matrix[startIndex][rightIndex]):
                flag = True
                return flag
        return flag

object = Solution()
output = object.searchMatrix(matrix, target)
print(output)
# x = np.arange(9).reshape(3,3)
# print(x)
# print(x[2])
# y = x[1:3,1:3]
# print(y)
# x = np.zeros([3,3])
#
# print(y)
# 数组的 dtype 为 int8（一个字节）
# a = [np.nan] * 4
# x = np.array(a, dtype=np.int8, order='F')
# x = a[2:3]
# print(a)
# b = np.isnan(a)
# print(b)
# c = ~np.isnan(a)
# print(c)
# y = x.reshape(3,2)
# print(y)

# print(x.itemsize)
#
# y = x.reshape(3,2)
# # 数组的 dtype 现在为 float64（八个字节）
# # y = np.array([1, 2, 3, 4, 5], dtype=np.float32)
# print(y)
#
# print(x)
# # print(y)