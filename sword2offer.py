from extra_class import *
from typing import *


class Solution:
    def __init__(self):
        pass

    def Find(self, target, array):
        if len(array) == 0:
            return False
        else:
            if len(array[0]) == 0:
                return False
        for i in range(len(array)):
            if target >= array[i][0] and target <= array[i][len(array[0]) - 1]:
                index = self.binary_search(array[i], target)
                if index != -1:
                    return True
        return False

    def binary_search(self, nums, target):
        low = 0
        high = len(nums) - 1
        while low <= high and low < len(nums) and high < len(nums):
            mid = int((low + high) / 2)
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                high = mid - 1
                mid = int((low + high) / 2)
            if nums[mid] < target:
                low = mid + 1
                mid = int((low + high) / 2)
        return -1

    def replaceSpace(self, s):
        ss = ""
        for i in s:
            if i == ' ':
                ss = ss + '%20'
            else:
                ss = ss + i
        return ss

    def printListFromTailToHead(self, listNode):
        if listNode == None:
            return []
        pre = None
        cur = listNode
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        res = []
        cur = pre
        while cur:
            res.append(cur.val)
            cur = cur.next
        return res

    def Fibonacci(self, n):
        # write code here
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 1
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def jumpFloorII(self, number):
        dp = [0] * (number + 1)
        dp[1] = 1
        for i in range(2, number + 1):
            dp[i] = 2 * dp[i - 1]
        return dp[number]

    def NumberOf1(self, n):
        count = 0
        if n < 0:
            count = 1
            n = abs(n)
        k = self.bin_helper(n)
        for i in k:
            if i == '1':
                count += 1
        return count

    def bin_helper(self, n):
        return bin(n)[2:]

    def Power(self, base, exponent):
        flag = 0
        if exponent % 2 == 1:
            exponent = exponent - 1
            flag = 1
        count = 0
        while exponent != 0:
            exponent = int(exponent / 2)
            count += 1
        res = 1
        for i in range(count):
            res = res * (base ** 2)
        if flag == 1:
            res = res * base
        return res

    def reOrderArray(self, array):
        x = []
        y = []
        for num in array:
            if num % 2 == 0:
                y.append(num)
            else:
                x.append(num)
        return x + y

    def FindKthToTail(self, head, k):
        fast = head
        slow = head
        if head == None or k <= 0:
            return None
        for i in range(1, k):
            if fast.next:
                fast = fast.next
            else:
                return None
        while fast.next:
            slow = slow.next
            fast = fast.next
        return slow

    def ReverseList(self, pHead):
        pre = None
        cur = pHead
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre

    def Merge(self, pHead1, pHead2):
        pre = ListNode(0)
        cur = pre
        while pHead1 and pHead2:
            if pHead1.val <= pHead2.val:
                cur.next = pHead1
                pHead1 = pHead1.next
                cur = cur.next
            else:
                cur.next = pHead2
                pHead2 = pHead2.next
                cur = cur.next
        if pHead1:
            cur.next = pHead1
        if pHead2:
            cur.next = pHead2
        return pre.next

    def HasSubtree(self, pRoot1, pRoot2):
        r = False
        if pRoot1 and pRoot2:
            if pRoot1.val == pRoot2.val:
                r = self.is_sub_tree_helper(pRoot1, pRoot2)
            if r == False:
                r = self.is_sub_tree_helper(pRoot1.left, pRoot2)
            if r == False:
                r = self.is_sub_tree_helper(pRoot1.right, pRoot2)
        return r

    def is_sub_tree_helper(self, node1, node2):
        if node2 == None:
            return True
        if node1 == None:
            return False
        if node1.val != node2.val:
            return False
        return self.is_sub_tree_helper(node1.left, node2.left) and self.is_sub_tree_helper(node1.right, node2.right)

    def Mirror(self, root):
        if root == None:
            return root
        tmp = root.left
        root.left = root.right
        root.right = tmp
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root

    def IsPopOrder(self, pushV, popV):
        stack = []
        j = 0
        for i in range(len(pushV)):
            stack.append(pushV[i])
            while j < len(popV) and stack[-1] == popV[j]:
                stack.pop()
                j += 1
        if len(stack) == 0:
            return True
        return False

    def PrintFromTopToBottom(self, root):
        if root == None:
            return []
        from queue import Queue
        q = Queue()
        q.put(root)
        l = []
        while q.empty() == False:
            size = q.qsize()
            for i in range(size):
                cur = q.get()
                l.append(cur.val)
                if cur.left:
                    q.put(cur.left)
                if cur.right:
                    q.put(cur.right)
        return l

    def VerifySquenceOfBST(self, sequence):
        root = sequence[-1]
        c = 0
        for i in range(len(sequence)):
            if sequence[i] > root:
                c = i
                break
        for j in range(c + 1, len(sequence)):
            if sequence[j] < root:
                return False
        left = True
        if i > 0:
            left = self.VerifySquenceOfBST(sequence[0:i])
        right = True
        if i < len(sequence) - 1:
            right = self.VerifySquenceOfBST(sequence[i + 1:])
        return left and right

    def FindPath(self, root, expectNumber):
        self.l = []
        self.path_helper(root, expectNumber, [])
        return self.l

    def path_helper(self, cur_node, cur_num, cur_path):
        if cur_node == None:
            return
        cur_path.append(cur_node.val)
        if cur_node.val == cur_num and cur_node.left == None and cur_node.right == None:
            import copy
            self.l.append(copy.deepcopy(cur_path))
        if cur_node.left:
            self.path_helper(cur_node.left, cur_num - cur_node.val, cur_path)
        if cur_node.right:
            self.path_helper(cur_node.right, cur_num - cur_node.val, cur_path)
        cur_path.pop()

    def MoreThanHalfNum_Solution(self, numbers):
        h = {}
        n = int(len(numbers) / 2) + 1
        for num in numbers:
            try:
                h[num] += 1
            except:
                h[num] = 1
            if h[num] >= n:
                return num
        return 0

    def FindGreatestSumOfSubArray(self, array):
        s = 0
        max_value = array[0]
        for i in range(len(array)):
            s = s + array[i]
            max_value = max(max_value, s)
            if s < 0:
                s = 0
        return max_value

    def GetUglyNumber_Solution(self, index):
        if (index <= 0):
            return 0
        uglyList = [1]
        indexTwo = 0
        indexThree = 0
        indexFive = 0
        for i in range(index - 1):
            newUgly = min(uglyList[indexTwo] * 2, uglyList[indexThree] * 3, uglyList[indexFive] * 5)
            uglyList.append(newUgly)
            if (newUgly % 2 == 0):
                indexTwo += 1
            if (newUgly % 3 == 0):
                indexThree += 1
            if (newUgly % 5 == 0):
                indexFive += 1
        return uglyList[-1]

    def FirstNotRepeatingChar(self, s):
        h = {}
        for char in s:
            try:
                h[char] += 1
            except:
                h[char] = 1
        r = []
        for key in h:
            if h[key] == 1:
                r.append(key)
        for i in range(len(s)):
            if s[i] in r:
                return i
        return -1

    def InversePairs(self, data):
        count = 0
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if data[i] > data[j]:
                    count += 1
        return count % 1000000007

    def FindFirstCommonNode(self, pHead1, pHead2):
        s1 = []
        s2 = []
        if pHead1 == None or pHead2 == None:
            return None
        while pHead1:
            s1.append(pHead1)
            pHead1 = pHead1.next
        while pHead2:
            s2.append(pHead2)
            pHead2 = pHead2.next
        pre_a = s1.pop()
        pre_b = s2.pop()
        if pre_a != pre_b:
            return None
        while len(s1) != 0 and len(s2) != 0:
            cur_a = s1.pop()
            cur_b = s2.pop()
            if cur_a != cur_b:
                return pre_a
            else:
                pre_a = cur_a
                pre_b = cur_b
        if pre_a != pre_b:
            return None
        return pre_a

    def GetNumberOfK(self, data, k):
        cur_index = self.binary_search(data, k)
        if cur_index == -1:
            return 0
        count = 1
        left_index = cur_index - 1
        right_index = cur_index + 1
        while left_index >= 0:
            if data[left_index] == k:
                count += 1
                left_index = left_index - 1
            else:
                break
        while right_index < len(data):
            if data[right_index] == k:
                count += 1
                right_index = right_index + 1
            else:
                break
        return count

    def binary_search(self, nums, target):
        low = 0
        high = len(nums) - 1
        while low <= high and low < len(nums) and high < len(nums):
            mid = int((low + high) / 2)
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                high = mid - 1
                mid = int((low + high) / 2)
            if nums[mid] < target:
                low = mid + 1
                mid = int((low + high) / 2)
        return -1

    def TreeDepth(self, pRoot):
        root = pRoot
        if root == None:
            return 0
        if root.left == None and root.right == None:
            return 1
        if root.left and root.right == None:
            return self.TreeDepth(root.left) + 1
        if root.right and root.left == None:
            return self.TreeDepth(root.right) + 1
        if root.left and root.right:
            return max(self.TreeDepth(root.left), self.TreeDepth(root.right)) + 1

    def IsBalanced_Solution(self, pRoot):
        root = pRoot
        if root == None:
            return True
        left = self.tree_depth(root.left)
        right = self.tree_depth(root.right)
        if abs(left - right) <= 1:
            if self.IsBalanced_Solution(root.left) and self.IsBalanced_Solution(root.right):
                return True
        return False

    def tree_depth(self, root):
        if root == None:
            return 0
        if root.left == None and root.right == None:
            return 1
        return max(self.tree_depth(root.left), self.tree_depth(root.right)) + 1

    def FindNumsAppearOnce(self, array):
        h = {}
        for num in array:
            try:
                h[num] += 1
            except:
                h[num] = 1
        k = []
        for key in h.keys():
            if h[key] == 1:
                k.append(key)
        return k[0], k[1]

    def FindContinuousSequence(self, tsum):
        self.l = []
        for i in range(1, tsum):
            self.helper(i, tsum, [])
        return self.l

    def helper(self, start_num, cur_num, cur_list):
        if cur_num == 0:
            import copy
            self.l.append(copy.deepcopy(cur_list))
        if cur_num < start_num:
            return
        cur_list.append(start_num)
        self.helper(start_num + 1, cur_num - start_num, cur_list)

    def FindNumbersWithSum(self, array, tsum):
        for i in range(len(array)):
            if array[i] > tsum:
                break
            target = tsum - array[i]
            for j in range(i + 1, len(array)):
                if array[j] == target:
                    return array[i], array[j]
        return

    def LeftRotateString(self, s, n):
        end = len(s)
        return s[n:end] + s[:n]

    def ReverseSentence(self, s):
        kk = s.split(" ")
        r = ""
        for i in range(len(kk) - 1, 0, -1):
            r = r + kk[i] + ' '
        r += kk[0]
        return r

    def IsContinuous(self, numbers):
        if len(numbers) != 5:
            return False
        numbers.sort()
        num_0 = numbers.count(0)
        cur_nums = numbers[num_0:]
        count = 0
        for i in range(len(cur_nums) - 1):
            if cur_nums[i] == cur_nums[i + 1]:
                return False
            count = count + (cur_nums[i + 1] - cur_nums[i] - 1)
            if count > num_0:
                return False
        if count <= num_0:
            return True

    def LastRemaining_Solution(self, n, m):
        if m < 1 and n < 1:
            return -1
        last = 0
        for i in range(2, n + 1):
            last = (last + m) % i
        return last

    def duplicate(self, numbers, duplication):
        h = {}
        for num in numbers:
            try:
                h[num]
                duplication[0] = num
                return True
            except:
                h[num] = 0
        return False

    def multiply(self, A):
        b1 = [1] * len(A)
        b2 = [1] * len(A)
        tmp = 1
        for i in range(1, len(A)):
            tmp = tmp * A[i - 1]
            b1[i] = tmp
        tmp = 1
        for i in range(len(A) - 2, -1, -1):
            tmp = tmp * A[i + 1]
            b2[i] = tmp
        B = [1] * len(A)
        for i in range(len(A)):
            B[i] = b1[i] * b2[i]
        return B

    def EntryNodeOfLoop(self, pHead):
        if pHead == None or pHead.next == None:
            return None
        slow = pHead
        fast = pHead
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                slow = pHead
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return slow
        return None

    def deleteDuplication(self, pHead):
        if pHead == None or pHead.next == None:
            return pHead
        pre = ListNode(0)
        cur = pHead
        pre.next = pHead
        node = pre
        while cur and cur.next:
            if cur.val != cur.next.val:
                node = cur
            else:
                while cur.next and cur.val == cur.next.val:
                    cur = cur.next
                    node.next = cur.next
            cur = cur.next
        return pre.next

    def isSymmetrical(self, pRoot):
        root = pRoot
        if root == None:
            return True
        return self.helper(root.left, root.right)

    def helper(self, left, right):
        if left == None and right == None:
            return True
        if left and right == None:
            return False
        if right and left == None:
            return False
        if right.val != left.val:
            return False
        return self.helper(left.left, right.right) and self.helper(left.right, right.left)

    def Print(self, pRoot):
        r = []
        from queue import Queue
        q = Queue()
        q.put(pRoot)
        while q.empty() == False:
            c = []
            for i in range(q.qsize()):
                cur = q.get()
                c.append(cur)
                if cur.left:
                    q.put(cur.left)
                if cur.right:
                    q.put(cur.right)
            import copy
            r.append(copy.deepcopy(c))
        return r

    def KthNode(self, pRoot, k):
        self.l = []
        self.helper(pRoot)
        if k <= 0 or len(self.l) < k:
            return None
        return self.l[k - 1]

    def helper(self, cur_node):
        if cur_node == None:
            return
        self.helper(cur_node.left)
        self.l.append(cur_node)
        self.helper(cur_node.right)

    def maxInWindows(self, num, size):
        r = []
        if size <= 0 or len(num) == 0:
            return []
        for i in range(len(num)):
            if i + size > len(num):
                break
            tmp = num[i:i + size]
            r.append(max(tmp))
        return r

    def movingCount(self, threshold, rows, cols):
        pass


if __name__ == '__main__':
    s = Solution()
    k = s.maxInWindows([2, 3, 4, 2, 6, 2, 5, 1], 3)
    print(k)
