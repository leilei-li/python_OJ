class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Mianshi:
    def __init__(self):
        pass

    def __swap(self, num, a, b):
        t = num[a]
        num[a] = num[b]
        num[b] = t

    # def binary_search(self, num, target):
    #     low = 0
    #     high = len(num) - 1
    #     mid = int((low + high) / 2)
    #     while low <= high and low <= len(num) - 1 and high <= len(num) - 1:
    #         if num[mid] < target:
    #             low = mid + 1
    #             mid = int((low + high) / 2)
    #         if num[mid] > target:
    #             high = mid - 1
    #             mid = int((low + high) / 2)
    #         if num[mid] == target:
    #             return mid
    #     return -1
    #
    # def quick_sort(self, num, low, high):
    #     l = low
    #     h = high
    #     p = num[low]
    #     while l < h:
    #         while num[h] >= p and l < h:
    #             h = h - 1
    #         if l < h:
    #             self.__swap(num, l, h)
    #             l = l + 1
    #         while num[l] <= p and l < h:
    #             l = l + 1
    #         if l < h:
    #             self.__swap(num, l, h)
    #             h = h - 1
    #     if l > low:
    #         self.quick_sort(num, low, l - 1)
    #     if h < high:
    #         self.quick_sort(num, l + 1, high)
    #     self.num = num

    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x < 0:
            flag = 1
            x = 0 - x
        else:
            flag = 0
        c = 0
        while x != 0:
            t = x % 10
            x = int(x / 10)
            c = c * 10 + t
        if flag == 1:
            c = 0 - c
            if c <= -(2 ** 31):
                c = 0
        if flag == 0:
            if c >= 2 ** 31 - 1:
                c = 0
        return c

    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        c = 0
        xx = x
        while x != 0:
            t = x % 10
            x = int(x / 10)
            c = c * 10 + t
        if c == xx:
            return True
        else:
            return False

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ''
        r = strs[0]
        for i in range(1, len(strs)):
            r = self.__longestCommonPrefix(r, strs[i])
            if r == '':
                return ''
        return r

    def __longestCommonPrefix(self, qian, b):
        s = ''
        for i in range(min(len(qian), len(b))):
            if qian[i] == b[i]:
                s += qian[i]
            else:
                break
        return s

    def findLCS(self, A, B):
        m = len(B)
        n = len(A)
        dp = [[0] * (m + 1)] * (n + 1)
        if n == 0 or m == 0:
            return 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[n][m]

    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        h = {}
        for i in range(len(nums)):
            if nums[i] in h.keys():
                return [h[nums[i]], i]
            else:
                k = target - nums[i]
                h[k] = i

    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        head = ListNode(0)
        cur = head
        temp = 0
        while l1 != None or l2 != None:
            if l1 != None:
                temp += l1.val
                l1 = l1.next
            if l2 != None:
                temp += l2.val
                l2 = l2.next
            cur.next = ListNode(temp % 10)
            cur = cur.next
            temp = int(temp / 10)
        if temp == 1:
            cur.next = ListNode(1)
            cur = cur.next
        return head.next

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        left_board = 0
        max_length = 0
        h = {}
        for i in range(len(s)):
            cur_right_board = 0
            if s[i] in h.keys():
                cur_right_board = h[s[i]] + 1
            left_board = max(left_board, cur_right_board)
            max_length = max(max_length, i - left_board + 1)
            h[s[i]] = i
        return max_length

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        s = nums1 + nums2
        s.sort()
        k = len(s)
        if k % 2 == 0:
            m = int(k / 2)
            return (s[m - 1] + s[m]) / 2
        else:
            m = int(k / 2)
            return s[m]

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        self.start_index = 0
        self.max_palind_len = 0
        if len(s) < 2:
            return s
        for i in range(len(s)):
            self.__find_palindrome(s, i, i)
            self.__find_palindrome(s, i, i + 1)
        return s[self.start_index:self.start_index + self.max_palind_len]

    def __find_palindrome(self, s, start, end):
        while start >= 0 and end < len(s) and s[start] == s[end]:
            start -= 1
            end += 1
            if self.max_palind_len < end - start - 1:
                self.start_index = start + 1
                self.max_palind_len = end - start - 1

    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        flag = 0
        if x < 0:
            flag = 1
        x = abs(x)
        t = 0
        while x != 0:
            t = t * 10 + x % 10
            x = int(x / 10)
        if flag == 1:
            if t > 2 ** 31:
                return 0
            return 0 - t
        else:
            if t > 2 ** 31 - 1:
                return 0
            return t

    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        start = 0
        for i in range(len(str)):
            if str[i] != " ":
                start = i
                break
        s = str[start:]
        if s == "":
            return 0
        index = 0
        flag = 1
        if s[0] == '-':
            flag = 0
            index += 1
        if s[0] == '+':
            flag = 1
            index += 1
        result = ""
        for i in range(index, len(s)):
            if s[i] >= "0" and s[i] <= "9":
                result += s[i]
            else:
                break
        if result == "":
            return 0
        result_int = int(result)
        if flag == 0:
            if result_int > 2 ** 31:
                return -2 ** 31
            else:
                return 0 - result_int
        if flag == 1:
            if result_int < 2 ** 31 - 1:
                return result_int
            else:
                return 2 ** 31 - 1

    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        t = 0
        pre_num = x
        while x != 0:
            t = t * 10 + x % 10
            x = int(x / 10)
        if t == pre_num:
            return True
        return False

    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        dp = [[False for i in range(len(p) + 1)] for j in range(len(s) + 1)]
        dp[0][0] = True
        for i in range(1, len(p) + 1):
            if p[i - 1] == '*' and i >= 2:
                dp[0][i] = dp[0][i - 2]
        for i in range(1, len(s) + 1):
            for j in range(1, len(p) + 1):
                if p[j - 1] == '.':
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*':
                    dp[i][j] = (dp[i][j - 1]  # *代表没有字符
                                or
                                dp[i][j - 2]  # *代表前面一个
                                or
                                (dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.'))
                                )
                else:
                    if s[i - 1] == p[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
        return dp[len(s)][len(p)]

    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        result = 0
        left = 0
        right = len(height) - 1
        while left < right:
            area = (right - left) * min(height[left], height[right])
            result = max(result, area)
            if height[left] > height[right]:
                right = right - 1
            else:
                left = left + 1
        return result

    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        fast_node = head
        low_node = head
        result = ListNode(0)
        result.next = head
        for i in range(n):
            fast_node = fast_node.next
        if fast_node == None:
            return low_node.next
        fast_node = fast_node.next
        while fast_node != None:
            low_node = low_node.next
            fast_node = fast_node.next
        low_node.next = low_node.next.next
        return result.next

    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        try:
            for i in range(len(s)):
                if s[i] == '(' or s[i] == '{' or s[i] == '[':
                    stack.append(s[i])
                    continue
                if (s[i] == ')' and stack[-1] == '(') or (s[i] == ']' and stack[-1] == '[') or (
                        s[i] == '}' and stack[-1] == '{'):
                    stack.pop()
                    continue
                else:
                    stack.append(s[i])
                    continue
        except:
            return False
        if len(stack) == 0:
            return True
        else:
            return False

    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        result = ListNode(0)
        cur = result
        while l1 != None and l2 != None:
            if l1.val < l2.val:
                cur.next = l1
                cur = cur.next
                l1 = l1.next
            else:
                cur.next = l2
                cur = cur.next
                l2 = l2.next
        if l1 == None:
            cur.next = l2
        elif l2 == None:
            cur.next = l1
        return result.next

    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        result = None
        for i in range(len(lists)):
            result = self.mergeTwoLists(result, lists[i])
        return result

    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        result = ListNode(0)
        result.next = head
        cur = result
        while cur.next != None and cur.next.next != None:
            p_0 = cur
            p_1 = cur.next
            p_2 = cur.next.next
            p_1.next = p_2.next
            p_2.next = p_1
            p_0.next = p_2
            cur = cur.next.next
        return result.next

    def ReverseList(self, pHead):
        """
        :type pHead: ListNode
        :rtype: ListNode
        """
        if pHead == None:
            return None
        if pHead.next == None:
            return pHead
        cur_pre = pHead
        cur = pHead.next
        cur_after = pHead.next.next
        while cur_after != None:
            cur.next = cur_pre
            cur_pre = cur
            cur = cur_after
            cur_after = cur_after.next
        cur.next = cur_pre  # after到底了，最后两个还需要换一次位置
        pHead.next = None  # 原来的头指针指向空
        return cur

    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        pHead = head
        if pHead == None:
            return None
        if pHead.next == None:
            return pHead
        cur_pre = pHead
        cur = pHead.next
        cur_after = pHead.next.next
        count = 0
        while count < k:
            cur.next = cur_pre
            cur_pre = cur
            cur = cur_after
            cur_after = cur_after.next
        cur_next = cur_pre
        pHead.next = cur_after.next
        return cur

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        count = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[count] = nums[i]
                count += 1
        return count

    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        count = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[count] = nums[i]
                count += 1
        return count

    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if len(needle) == 0:
            return 0
        if len(needle) > len(haystack):
            return -1
        for i in range(len(haystack)):
            if haystack[i] == needle[0] and len(haystack) - i > len(needle):
                flag = 1
                for j in range(1, len(needle)):
                    if haystack[i + j] != needle[j]:
                        flag = 0
                        break
                if flag == 0:
                    continue
                else:
                    return i
        return -1

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        low = 0
        high = len(nums) - 1
        if len(nums) == 1:
            if nums[0] == target:
                return 0
            else:
                return -1
        while low <= high:
            mid = low + int((high - low) / 2)
            if nums[mid] == target:
                return mid
            if nums[mid] < nums[high]:
                if nums[mid] < target and nums[high] >= target:
                    low = mid + 1
                else:
                    high = mid - 1
            if nums[mid] > nums[low]:
                if nums[mid] > target and nums[low] <= target:
                    high = mid - 1
                else:
                    low = mid + 1
        return -1


if __name__ == '__main__':
    m = Mianshi()
    num = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # print(m.lengthOfLongestSubstring('a'))
    # print(m.findMedianSortedArrays([1, 2], [3, 4]))
    # print(m.longestPalindrome("cbbd"))
    # print(m.isMatch("aa", "a"))
    # print(m.isMatch_2("aa","a"))
    # print(m.isValid("()[]{}"))
    print(m.search([1,3], 2))
