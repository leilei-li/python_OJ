from extra_class import *
from typing import *


class Solution:
    def __init__(self):
        pass

    def twoSum(self, nums, target):
        d = {}
        for i in range(len(nums)):
            cur_num = nums[i]
            temp = target - nums[i]
            if cur_num in d.keys():
                return [d[cur_num], i]
            else:
                d[temp] = i
        return None

    def addTwoNumbers(self, l1, l2):
        result_list_node = ListNode(0)
        return_node = result_list_node
        carry = 0
        while l1 != None or l2 != None:
            sum = 0
            if l1 != None:
                sum = l1.val
                l1 = l1.next
            if l2 != None:
                sum += l2.val
                l2 = l2.next
            sum += carry
            result_list_node.next = ListNode(sum % 10)
            result_list_node = result_list_node.next
            if sum >= 10:
                carry = 1
            else:
                carry = 0
        if carry == 1:
            result_list_node.next = ListNode(1)
        return return_node.next

    # def lengthOfLongestSubstring(self, s):

    def longestPalindrome(self, s):
        self.length = 0
        self.start_index = 0
        if len(s) < 2:
            return s
        for i in range(len(s)):
            self.longestPalindrome_helper(s, i, i)
            self.longestPalindrome_helper(s, i, i + 1)
        return s[self.start_index:self.start_index + self.length]

    def longestPalindrome_helper(self, s, start, end):
        """
        从zui开始的start==end(奇数),start==end+1(偶数)，
        开始向外扩散，如果左右两边扩散的字符相等，那么长度会+2
        :param s:
        :param start:
        :param end:
        :return:
        """
        while start >= 0 and end < len(s) and s[start] == s[end]:
            start = start - 1
            end = end + 1
        cur_length = end - start - 1
        if self.length < cur_length:
            self.length = cur_length
            self.start_index = start + 1  # 跳出循环时已经不是合法的start了

    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        dp = []
        for i in range(len(s) + 1):
            dp.append(False)
        dp[len(s)] = True
        i = len(p) - 1
        while i >= 0:
            if p[i] == '*':
                for j in range(len(s) - 1, -1, -1):
                    dp[j] = dp[j] or (dp[j + 1] and (p[i - 1] == '.' or p[i - 1] == s[j]))
                i = i - 1
            else:
                for j in range(len(s)):
                    dp[j] = dp[j + 1] and (p[i] == '.' or p[i] == s[j])
                dp[len(s)] = False
            i = i - 1
        return dp[0]

    def maxArea(self, height):
        left = 0
        right = len(height) - 1
        max_value = 0
        while right > left:
            cur = min(height[left], height[right]) * (right - left)
            max_value = max(max_value, cur)
            if height[left] > height[right]:
                right = right - 1
            elif height[left] <= height[right]:
                left = left + 1
        return max_value

    def threeSum(self, nums):
        result = []
        nums.sort()
        for i in range(len(nums)):
            twoSum = 0 - nums[i]
            cur_nums_list = nums[i + 1:]
            r = self.threeSum_twoSum_helper(cur_nums_list, twoSum)
            for rr in r:
                cur_list = [nums[i]] + rr
                if cur_list not in result:
                    result.append(cur_list)
        return result

    def threeSum_twoSum_helper(self, nums, target):
        d = {}
        result = []
        for i in range(len(nums)):
            tmp = target - nums[i]
            if nums[i] in d.keys():
                result.append([nums[d[nums[i]]], nums[i]])
            else:
                d[tmp] = i
        return result

    def removeNthFromEnd(self, head, n):
        pre_head = ListNode(0)
        pre_head.next = head
        fast = head
        for i in range(n):
            fast = fast.next
        low_pre = pre_head
        low_cur = head
        while fast:
            fast = fast.next
            low_pre = low_pre.next
            low_cur = low_cur.next
        low_pre.next = low_cur.next
        return pre_head.next

    def isValid(self, s):
        stack = []
        for char in s:
            try:
                if char == '(' or char == '{' or char == '[':
                    stack.append(char)
                if char == ')':
                    p = stack.pop()
                    if p != '(':
                        return False
                if char == '}':
                    p = stack.pop()
                    if p != '{':
                        return False
                if char == ']':
                    p = stack.pop()
                    if p != '[':
                        return False
            except:
                return False
        if len(stack) == 0:
            return True
        return False

    def mergeTwoLists(self, l1: ListNode, l2: ListNode):
        pre_head = ListNode(0)
        cur = ListNode(0)
        pre_head.next = cur
        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = ListNode(l1.val)
                cur = cur.next
                l1 = l1.next
            else:
                cur.next = ListNode(l2.val)
                cur = cur.next
                l2 = l2.next
        if l1:
            cur.next = l1
        if l2:
            cur.next = l2
        return pre_head.next.next

    def generateParenthesis(self, n: int):
        result = ['']
        for i in range(n * 2):
            mm = []
            for s in result:
                mm.append(s + '(')
                mm.append(s + ')')
            result = mm
        r = []
        for i in result:
            if self.generateParenthesis_helper(i):
                r.append(i)
        return r

    def generateParenthesis_helper(self, s):
        stack = []
        for char in s:
            try:
                if char == '(':
                    stack.append(char)
                if char == ')':
                    p = stack.pop()
                    if p != '(':
                        return False
            except:
                return False
        if len(stack) == 0:
            return True
        return False

    def mergeKLists(self, lists):
        if len(lists) == 0:
            return None
        if len(lists) == 1:
            return lists[0]
        if len(lists) == 2:
            return self.mergeKLists_helper(lists[0], lists[1])
        result = self.mergeKLists_helper(lists[0], lists[1])
        for i in range(2, len(lists)):
            result = self.mergeKLists_helper(result, lists[i])
        return result

    def mergeKLists_helper(self, l1: ListNode, l2: ListNode):
        pre_head = ListNode(0)
        cur = ListNode(0)
        pre_head.next = cur
        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = ListNode(l1.val)
                cur = cur.next
                l1 = l1.next
            else:
                cur.next = ListNode(l2.val)
                cur = cur.next
                l2 = l2.next
        if l1:
            cur.next = l1
        if l2:
            cur.next = l2
        return pre_head.next.next

    def longestValidParentheses(self, s):
        stack = []
        if len(s) == 0:
            return 0
        length = 0
        last = -1  # 对应的有效长度的'('的开始位置
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)  # 将所有'('位置存在栈中
            if s[i] == ')':
                if len(stack) == 0:
                    last = i
                else:
                    stack.pop()
                    if len(stack) == 0:
                        length = max(length, i - last)
                    else:
                        length = max(length, i - stack[-1])
        return length

    def search(self, nums, target):
        k = 0
        for i in range(1, len(nums)):
            if nums[i - 1] > nums[i]:
                k = i - 1
                # print(k)
                break
        try:
            k1 = self.binary_search(nums, 0, k, target)
            k2 = self.binary_search(nums, k + 1, len(nums) - 1, target)
        except:
            return -1
        if k1 == -1 and k2 == -1:
            return -1
        else:
            return max(k1, k2)

    def binary_search(self, nums, low, height, target):
        # print(nums[low:height])
        while low <= height:
            mid = int((low + height) / 2)
            # print(low, height, mid)
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                height = mid - 1
            if nums[mid] < target:
                low = mid + 1
        return -1

    def searchRange(self, nums, target):
        index = self.binary_search(nums, 0, len(nums) - 1, target)
        if index == -1:
            return [-1, -1]
        start = index
        end = index
        while start > 0:
            if nums[start - 1] == target:
                start = start - 1
                continue
            else:
                break
        while end < len(nums) - 1:
            if nums[end + 1] == target:
                end = end + 1
                continue
            else:
                break
        return [start, end]

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        self.result = []
        self.l = []
        self.__back_tacking_sum(candidates, 0, target)
        return self.result

    def __back_tacking_sum(self, candidates, position, target):
        if target == 0:
            self.result.append(self.l)
            return
        else:
            for i in range(position, len(candidates)):
                if candidates[i] > target:
                    return
                else:
                    self.l.append(candidates[i])
                    self.__back_tacking_sum(candidates, i, target - candidates[i])
                    self.l = self.l[:(len(self.l) - 1)]

    def trap(self, height):
        if len(height) <= 1:
            return 0
        highest_index = height.index(max(height))
        area = 0
        cur = height[0]
        for i in range(highest_index):
            if height[i] > cur:
                cur = height[i]
            else:
                area = area + (cur - height[i])
        cur = height[-1]
        for i in range(len(height) - 1, highest_index, -1):
            if height[i] > cur:
                cur = height[i]
            else:
                area = area + (cur - height[i])
        return area

    def groupAnagrams(self, strs):
        d = {}
        for str in strs:
            k = "".join(sorted(str))
            if k in d.keys():
                d[k].append(str)
            else:
                d[k] = [str]
        result = []
        for key in d.keys():
            result.append(d[key])
        return result

    def maxSubArray(self, nums):
        max_value = nums[0]
        sum = 0
        for i in range(len(nums)):
            sum = sum + nums[i]
            max_value = max(sum, max_value)
            if sum < 0:
                sum = 0
        return max_value

    def canJump(self, nums):
        dp = [False] * len(nums)
        dp[0] = True
        for i in range(len(nums)):
            if dp[i] == True:
                for j in range(1, nums[i] + 1):
                    if i + j < len(nums):
                        dp[i + j] = True
                    else:
                        return True
        return dp[len(nums) - 1]

    def merge(self, intervals):
        if len(intervals) <= 1:
            return intervals
        intervals.sort()
        result = []
        result.append(intervals[0])
        for index in range(1, len(intervals)):
            cur_start = intervals[index][0]
            cur_end = intervals[index][-1]
            result_end = result[-1][-1]
            if cur_start <= result_end:
                if result_end >= cur_end:
                    continue
                result[-1][-1] = cur_end
            else:
                result.append(intervals[index])
        return result

    def uniquePaths(self, m, n):
        dp = [[0] * n for i in range(m)]
        for i in range(n):
            dp[0][i] = 1
        for j in range(m):
            dp[j][0] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]

    def minPathSum(self, grid):
        m = len(grid)
        n = len(grid[0])
        dp = [[0] * n for i in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = grid[i][0] + dp[i - 1][0]
        for j in range(1, n):
            dp[0][j] = grid[0][j] + dp[0][j - 1]
        for i in range(1, m):
            for j in range(1, n):
                min_path = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
                dp[i][j] = min_path
        return dp[m - 1][n - 1]

    def climbStairs(self, n):
        if n == 1:
            return 1
        if n == 2:
            return 2
        dp = [0] * n
        dp[0] = 1
        dp[1] = 2
        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n - 1]

    def minDistance(self, word1, word2):
        m = len(word1)
        n = len(word2)
        if m == 0 or n == 0:
            return max(m, n)
        dp = [[0] * (n + 1) for i in range(m + 1)]
        for i in range(n + 1):
            dp[0][i] = i
        for i in range(m + 1):
            dp[i][0] = i

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return dp[m][n]

    def subsets(self, nums):
        if len(nums) == 0:
            return nums
        self.result = []
        cur_list = []
        nums.sort()
        self.subsets_helper(nums, 0, cur_list)
        return self.result

    def subsets_helper(self, nums, start_index, cur_list):
        import copy
        self.result.append(copy.deepcopy(cur_list))
        for i in range(start_index, len(nums)):
            cur_list.append(nums[i])
            # print(cur_list)
            self.subsets_helper(nums, i + 1, cur_list)
            cur_list.pop()

    def largestRectangleArea(self, heights):
        result = 0
        stack = []
        for i in range(len(heights)):
            if len(stack) == 0 or stack[-1] <= heights[i]:
                stack.append(heights[i])
            else:
                pop_count = 0
                while len(stack) != 0 and stack[-1] > heights[i]:
                    pop_count += 1
                    result = max(result, pop_count * stack[-1])
                    stack.pop()
                while pop_count >= 0:
                    stack.append(heights[i])
                    pop_count = pop_count - 1
        count = 1
        print(stack)
        while len(stack) != 0:
            result = max(result, stack.pop() * count)
            count += 1
        return result

    def partition(self, head: ListNode, x: int):
        small_node = ListNode(0)
        pre_node = small_node
        big_node = ListNode(0)
        connect = big_node
        cur_node = head
        while cur_node:
            if cur_node.val < x:
                small_node.next = ListNode(cur_node.val)
                small_node = small_node.next
            else:
                big_node.next = ListNode(cur_node.val)
                big_node = big_node.next
            cur_node = cur_node.next
        small_node.next = connect.next
        return pre_node.next

    def inorderTraversal(self, root: TreeNode) -> List[int]:
        self.l = []
        if root == None:
            return root
        else:
            self.inorder_helper(root)
        return self.l

    def inorder_helper(self, cur_node):
        if cur_node == None:
            return
        else:
            self.inorder_helper(cur_node.left)
            self.l.append(cur_node.val)
            self.inorder_helper(cur_node.right)

    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        return dp[n]

    def isValidBST(self, root: TreeNode) -> bool:
        self.l = []
        self.isValidBST_helper(root)
        for i in range(1, len(self.l)):
            if self.l[i - 1] >= self.l[i]:
                return False
        return True

    def isValidBST_helper(self, cur_node):
        if cur_node == None:
            return
        else:
            self.isValidBST_helper(cur_node.left)
            self.l.append(cur_node.val)
            self.isValidBST_helper(cur_node.right)

    def isSymmetric(self, root: TreeNode) -> bool:
        if root == None:
            return True
        return self.is_Symmetric(root.left, root.right)

    def is_Symmetric(self, cur_node_left, cur_node_right):
        if cur_node_left == None and cur_node_right == None:
            return True
        if cur_node_left == None and cur_node_right != None:
            return False
        if cur_node_left != None and cur_node_right == None:
            return False
        if cur_node_left.val != cur_node_right.val:
            return False
        return self.is_Symmetric(cur_node_left.left, cur_node_right.right) and self.is_Symmetric(cur_node_left.right,
                                                                                                 cur_node_right.left)

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root == None:
            return []
        from queue import Queue
        q = Queue()
        q.put(root)
        result = []
        while q.empty() == False:
            list = []
            for i in range(q.qsize()):
                cur_node = q.get()
                if cur_node.left:
                    q.put(cur_node.left)
                if cur_node.right:
                    q.put(cur_node.right)
                list.append(cur_node.val)
            result.append(list)
        return result

    def maxDepth(self, root: TreeNode) -> int:
        self.max_depth = 0
        if root == None:
            return self.max_depth
        else:
            self.max_depth_helper(root, 1)
        return self.max_depth

    def max_depth_helper(self, cur_node, cur_depth):
        if cur_node == None:
            return
        else:
            self.max_depth = max(self.max_depth, cur_depth)
            self.max_depth_helper(cur_node.left, cur_depth + 1)
            self.max_depth_helper(cur_node.right, cur_depth + 1)

    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root == None:
            return
        else:
            self.flatten(root.left)
            self.flatten(root.right)
            tmp = root.right
            root.right = root.left
            root.left = None
            while root.right != None:
                root = root.right
            root.right = tmp

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        max_profit = 0
        min_value = prices[0]
        for i in range(len(prices)):
            min_value = min(prices[i], min_value)
            max_profit = max(max_profit, prices[i] - min_value)
        return max_profit

    def maxPathSum(self, root: TreeNode) -> int:
        self.max_value = -9999
        self.max_path_sum_helper(root)
        return self.max_value

    def max_path_sum_helper(self, cur_node):
        if cur_node == None:
            return 0
        else:
            left = max(0, self.max_path_sum_helper(cur_node.left))
            right = max(0, self.max_path_sum_helper(cur_node.right))
            self.max_value = max(self.max_value, left + right + cur_node.val)
            return max(left, right) + cur_node.val

    def longestConsecutive(self, nums: List[int]) -> int:
        max_value = 0
        h = {}
        for num in nums:
            h[num] = 0
        for num in nums:
            low = num - 1
            high = num + 1
            count = 1
            while low in h:
                count += 1
                low = low - 1
            while high in h:
                count += 1
                high = high + 1
            max_value = max(max_value, count)
        return max_value

    def singleNumber(self, nums: List[int]) -> int:
        cur = nums[0]
        for i in range(1, len(nums)):
            cur = cur ^ nums[i]
        return cur

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for i in range(len(s) + 1):
            for j in range(0, i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        return dp[len(s)]

    def hasCycle(self, head):
        fast = head
        low = head
        try:
            fast = fast.next.next
        except:
            return False
        while fast:
            if fast == low:
                return True
            try:
                fast = fast.next.next
            except:
                return False
            low = low.next
        return False

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast = head
        slow = head
        if head == None or head.next == None:
            return None
        while fast.next != None and fast.next.next != None:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        return None

    def maxProduct(self, nums: List[int]) -> int:
        max_value = -9999
        imax = 1
        imin = 1
        for i in range(len(nums)):
            if nums[i] < 0:
                tmp = imax
                imax = imin
                imin = tmp
            imax = max(nums[i] * imax, nums[i])
            imin = min(nums[i] * imin, nums[i])
            max_value = max(imax, max_value)
        return max_value

    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        stack_a = []
        stack_b = []
        if headA == None or headB == None:
            return None
        while headA:
            stack_a.append(headA)
            headA = headA.next
        while headB:
            stack_b.append(headB)
            headB = headB.next
        pre_a = stack_a.pop()
        pre_b = stack_b.pop()
        if pre_a != pre_b:
            return None
        else:
            for i in range(min(len(stack_a), len(stack_b))):
                cur_a = stack_a.pop()
                cur_b = stack_b.pop()
                if cur_a != cur_b:
                    return pre_a
                else:
                    pre_a = cur_a
                    pre_b = cur_b
            if pre_a != pre_b:
                return None
            else:
                return pre_a

    def majorityElement(self, nums: List[int]) -> int:
        h = {}
        for num in nums:
            if num in h:
                h[num] = h[num] + 1
            else:
                h[num] = 1
            if h[num] > int(len(nums) / 2):
                return num

    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[len(nums) - 1]

    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        cur = head
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre

    def findKthLargest(self, nums: List[int], k: int) -> int:
        max_value = max(nums)
        min_value = min(nums)
        n = max_value - min_value
        bucket = [0] * (n + 1)
        for num in nums:
            bucket[num - min_value] += 1
        for i in range(n, -1, -1):
            if bucket[i] > 0:
                k = k - bucket[i]
            if k <= 0:
                return i + min_value
        return 0

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        if m < 1:
            return 0
        n = len(matrix[0])
        dp = [[0] * (n + 1) for i in range(m + 1)]
        max_value = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if matrix[i - 1][j - 1] == '1':
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                    max_value = max(max_value, dp[i][j])
        return max_value ** 2

    def invertTree(self, root: TreeNode) -> TreeNode:
        if root == None:
            return root
        else:
            self.invert_tree_helper(root)
        return root

    def invert_tree_helper(self, cur_node):
        if cur_node == None:
            return
        else:
            self.invert_tree_helper(cur_node.left)
            self.invert_tree_helper(cur_node.right)
            tmp = cur_node.left
            cur_node.left = cur_node.right
            cur_node.right = tmp

    def isPalindrome(self, head: ListNode) -> bool:
        if head == None:
            return True
        else:
            fast = head
            slow = head
            while fast.next and fast.next.next:
                fast = fast.next.next
                slow = slow.next
            new_head = self.reversed_helper(slow)
            node_a = head
            node_b = new_head
            while node_a:
                if node_a.val != node_b.val:
                    return False
                node_a = node_a.next
                node_b = node_b.next
            return True

    def reversed_helper(self, cur_node):
        pre = None
        cur = cur_node
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root == None:
            return None
        if root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)  # 判断左子树有没有p，q
        right = self.lowestCommonAncestor(root.right, p, q)  # 判断右子树有没有p,q
        if left and right:
            return root
        if left and right == None:
            return left
        if right and left == None:
            return right
        return None

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        left_mul = [1] * len(nums)
        right_mul = [1] * len(nums)
        cur_result = 1
        for i in range(1, len(nums)):
            left_mul[i] = cur_result * nums[i - 1]
            cur_result = left_mul[i]
        cur_result = 1
        for i in range(len(nums) - 2, -1, -1):
            right_mul[i] = cur_result * nums[i + 1]
            cur_result = right_mul[i]
        result = []
        for i in range(len(nums)):
            result.append(left_mul[i] * right_mul[i])
        return result

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        cur_index = 0
        result = []
        if len(nums) == 0:
            return []
        while cur_index + k <= len(nums):
            result.append(max(nums[cur_index:cur_index + k]))
            cur_index += 1
        return result

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix) == 0:
            return False
        if len(matrix[0]) == 0:
            return False
        for i in range(len(matrix)):
            cur_list = matrix[i]
            if target >= cur_list[0] and target <= cur_list[len(cur_list) - 1]:
                index = self.binary_search_helper(cur_list, target)
                if index:
                    return True
        return False

    def binary_search_helper(self, nums, target):
        low = 0
        high = len(nums) - 1
        mid = int((low + high) / 2)
        while low <= high:
            if nums[mid] == target:
                return True
            if nums[mid] > target:
                high = mid - 1
                mid = int((low + high) / 2)
            if nums[mid] < target:
                low = mid + 1
                mid = int((low + high) / 2)
        return None

    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums)):
            if nums[i] == 0:
                not_zero_index = self.find_next_index_not_zero(nums, i)
                if not_zero_index == -1:
                    continue
                nums[i] = nums[not_zero_index]
                nums[not_zero_index] = 0

        print(nums)

    def find_next_index_not_zero(self, nums, start_index):
        for i in range(start_index + 1, len(nums)):
            if nums[i] != 0:
                return i
        return -1

    def findDuplicate(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            cur_num = abs(nums[i])
            if nums[cur_num] < 0:
                return cur_num
            else:
                nums[cur_num] = -nums[cur_num]

    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        if len(prices) <= 1:
            return 0
        dayprofit = [0] * len(prices)
        for i in range(1, len(prices)):
            if prices[i - 1] < prices[i]:
                dayprofit[i] = prices[i] - prices[i - 1]
        for i in range(1, len(prices)):
            if dayprofit[i] > 0:
                profit += dayprofit[i]
        return profit

    def maxCoins(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        if len(nums) < 2:
            return nums[0]
        nums = [1] + nums + [1]
        dp = [[0] * len(nums) for i in range(len(nums))]
        for i in range(len(nums) - 1, -1, -1):
            for j in range(i + 2, len(nums)):
                for k in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])
        return dp[0][len(nums) - 1]

    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [9999] * (amount + 1)
        dp[0] = 0
        for i in range(amount + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        if dp[amount] == 9999:
            return -1
        else:
            return dp[amount]

    def countBits(self, num: int) -> List[int]:
        dp = [0] * (num + 1)
        for i in range(int(num / 2) + 1):
            dp[2 * i] = dp[i]
            if 2 * i + 1 <= num:
                dp[2 * i + 1] = dp[i] + 1
        return dp

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        h = {}
        for num in nums:
            try:
                h[num] = h[num] + 1
            except:
                h[num] = 1
        bucket = {}
        max_v = 0
        for key in h.keys():
            v = h[key]
            max_v = max(max_v, v)
            if v in bucket.keys():
                bucket[v].append(key)
            else:
                bucket[v] = [key]
        count = 0
        res = []
        for i in range(max_v, -1, -1):
            if i in bucket.keys():
                count = count + len(bucket[i])
                res = res + bucket[i]
                if count >= k:
                    return res

    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people = sorted(people, key=lambda x: (-x[0], x[1]))
        result = []
        for each in people:
            result.insert(each[1], each)
        return result

    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        c = sum(nums)
        if n == 0:
            return False
        if c % 2 == 1:
            return False
        dp = [[0] * (c + 1) for i in range(n)]
        for i in range(c + 1):
            if nums[0] != i:
                dp[0][i] = False
            else:
                dp[0][i] = True
        for i in range(1, n):
            for j in range(c + 1):
                dp[i][j] = dp[i - 1][j]
                if nums[i] <= j:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i]]
        return dp[n - 1][c]

    def pathSum(self, root: TreeNode, sum: int) -> int:
        if root == None:
            return number
        self.path_sum_helper(root, sum)
        self.pathSum(root.left, sum)
        self.pathSum(root.right, sum)
        return self.number

    def path_sum_helper(self, cur_node, cur_sum):
        if cur_node == None:
            return
        cur_sum = cur_sum - cur_node.val
        if cur_sum == 0:
            self.number += 1
            return
        else:
            self.path_sum_helper(cur_node.left, cur_sum)
            self.path_sum_helper(cur_node.right, cur_sum)

    def findAnagrams(self, s: str, p: str) -> List[int]:
        n = len(p)
        i = 0
        result = []
        while i + n <= len(s):
            ss = s[i:i + n]
            if self.is_anagrams(ss, p):
                result.append(i)
            i = i + 1
        return result

    def is_anagrams(self, a, b):
        h_a = {}
        h_b = {}
        for char in a:
            try:
                h_a[char] += 1
            except:
                h_a[char] = 1
        for char in b:
            try:
                h_b[char] += 1
            except:
                h_b[char] = 1
        for k in h_a.keys():
            try:
                if h_a[k] != h_b[k]:
                    return False
            except:
                return False
        return True

    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        dp = [0] * (len(nums) + 1)
        for num in nums:
            dp[num] = 1
        r = []
        for i in range(1, len(nums) + 1):
            if dp[i] == 0:
                r.append(i)
        return r

    def hammingDistance(self, x: int, y: int) -> int:
        x = self.int_to_binary(x)
        y = self.int_to_binary(y)
        len_x = len(x)
        len_y = len(y)
        if len_x > len_y:
            y = "0" * (len_x - len_y) + y
        if len_y > len_x:
            x = "0" * (len_y - len_x) + x
        count = 0
        for i in range(len(x)):
            if x[i] != y[i]:
                count += 1
        return count

    def int_to_binary(self, x):
        b = bin(x)[2:]
        return b

    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        sum1 = sum(nums)
        if sum1 < S or (sum1 + S) % 2 == 1:
            return 0
        w = int((sum1 + S) / 2)
        dp = [0] * (w + 1)
        dp[0] = 1
        for num in nums:
            for j in range(w, num - 1, -1):
                dp[j] = dp[j] + dp[j - num]
        return dp[w]

    def convertBST(self, root: TreeNode) -> TreeNode:
        def getAllTree(r: TreeNode) -> [TreeNode]:
            if r is None:
                return []
            return getAllTree(r.right) + [r] + getAllTree(r.left)

        temp = getAllTree(root)
        for idx in range(len(temp)):
            if idx >= 1:
                temp[idx].val += temp[idx - 1].val

        return root

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.max_depth = 0
        self.depth_helper(root)
        return self.max_depth

    def depth_helper(self, cur_node):
        if cur_node != None:
            right = self.depth_helper(cur_node.right)
            left = self.depth_helper(cur_node.left)
            self.max_depth = max(self.max_depth, right + left)
            return max(right, left) + 1
        else:
            return 0

    def subarraySum(self, nums: 'List[int]', k: 'int') -> 'int':
        sum, res, cul = 0, 0, {}
        cul[0] = 1
        for i in range(len(nums)):
            sum += nums[i]
            if sum - k in cul:
                res += cul[sum - k]
            if sum not in cul:
                cul[sum] = 0
            cul[sum] += 1
        return res

    def findUnsortedSubarray(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        max_value = -9999
        min_value = 9999
        l = 0
        r = 0
        for i in range(len(nums)):
            if nums[i] < max_value:
                r = i
            max_value = max(max_value, nums[i])
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] > min_value:
                l = i
            min_value = min(min_value, nums[i])
        if l == r:
            return 0
        else:
            return r - l + 1

    # def countSubstrings(self, s: str) -> int:
    #     count = 0
    #     for i in range(len(s)):
    #         for j in range(i + 1, len(s) + 1):
    #             if self.palin_helper(s, i, j):
    #                 # print(s[i:j])
    #                 count += 1
    #     return count
    #
    # def palin_helper(self, s, start_index, end_index):
    #     a = []
    #     b = []
    #     s = s[start_index:end_index]
    #     for i in range(len(s)):
    #         a.append(s[i])
    #         b.append(s[i])
    #     for i in range(len(a)):
    #         a1 = a[i]
    #         b1 = b.pop()
    #         if a1 != b1:
    #             return False
    #     return True

    def countSubstrings(self, s: str) -> int:
        self.num = 0
        for i in range(len(s)):
            self.count_helper(s, i, i)
            self.count_helper(s, i, i + 1)
        return self.num

    def count_helper(self, s, start, end):
        while start >= 0 and end < len(s) and start <= end and s[start] == s[end]:
            self.num += 1
            start = start - 1
            end = end + 1

    def dailyTemperatures(self, T: List[int]) -> List[int]:
        r = []
        for i in range(len(T)):
            flag = 0
            for j in range(i + 1, len(T)):
                if T[j] > T[i]:
                    r.append(j - i)
                    flag = 1
                    break
            if flag == 0:
                r.append(0)
        return r


if __name__ == '__main__':
    s = Solution()
    k = s.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73])
    print(k)
