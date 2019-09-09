class BestTimeToBuyAndSellStock:
    def maxProfit(self, prices):
        min_price = float('inf')
        max_profit = 0
        for i in range(len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
            else:
                max_profit = max(max_profit, prices[i]-min_price)
        return max_profit

class BestTimeToBuyAndSellStockII:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                max_profit += prices[i] - prices[i-1]
        return max_profit

class BinaryTreeInorderTraversal:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        self.inorder(res, root)
        return res
    
    def inorder(self, res, root):
        if root:
            self.inorder(self, res, root.left)
            res.append(root.val)
            self.inorder(self, res, root.right)

class BinaryTreeLevelOrderTraversal:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        level = [root]
        res = []
        while root and level:
            nextLevel = []
            currNodes = []
            for node in level:
                res.append(node.val)
                if node.left:
                    nextLevel.append(node.left)
                if node.right:
                    nextLevel.append(node.right)
            level = nextLevel
            res.append(currNodes)
        return res

# Use Level Order Traversal
class BinaryTreeZigzagLevelOrderTraversal:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        level = [root]
        height = 0
        while root and level:
            currNodes = []
            nextLevel = []
            for node in level:
                currNodes.append(node.val)
                if node.left:
                    nextLevel.append(node.left)
                if node.right:
                    nextLevel.append(node.right)
            level = nextLevel
            if height%2:
                res.append(currNodes[::-1])
            else:
                res.append(currNodes)
            height += 1
        return res

class CourseSchedule:
    #Topological Sort
    def canFinish(self, numCourses: int, edges: List[List[int]]) -> bool:   
        vertices = {x:[] for x in range(numCourses)}
        indegrees = {x:0 for x in range(numCourses)}
        for edge in edges:
            vertices[edge[0]].append(edge[1])
            indegrees[edge[1]] += 1
        while vertices and edges:
            root = min(indegrees, key=indegrees.get)
            if indegrees[root]:
                return False
            for child in vertices[root]:
                indegrees[child] -= 1
            vertices.pop(root)
            indegrees.pop(root)
        return True

class CourseScheduleII:
    def findOrder(self, numCourses: int, edges: List[List[int]]) -> List[int]:
        if not edges:
            return [x for x in range(numCourses)]
        vertices = {x:[] for x in range(numCourses)}
        indegrees = {x:0 for x in range(numCourses)}
        res = []
        for edge in edges:
            vertices[edge[0]].append(edge[1])
            indegrees[edge[1]] += 1
        while vertices and edges:
            root = min(indegrees, key=indegrees.get)
            if indegrees[root]:
                return []
            for child in vertices[root]:
                indegrees[child] -= 1
            res.append(root)
            vertices.pop(root)
            indegrees.pop(root)
        return res[::-1]

class ClimbingStairs:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        first, second = 1, 2
        for i in range(3, n+1):
            third = first + second
            first, second = second, third
        return second


    def fib(self, n: int) -> int:
        first, second = 0, 1
        for i in range(n):
            first, second = second, first+second
        return first

class ContainsDuplicate:
    def containsDuplicate(self, nums: List[int]) -> bool:
        _set = set()
        for i in nums:
            if i in _set:
                return True
            _set.add(i)
        return False

class ConvertSortedArraytoBinarySearchTree:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root

class DeleteNodeinaLinkedList:
    def deleteNode(self, node):
        node.val = node.next.val
        node.next = node.next.next

class ExcelSheetColumnNumber:
    def titleToNumber(self, s: str) -> int:
        res = 0
        for i in range(len(s)):
            res = res*26 + ord(s[i])-64
        return res

class FactorialTrailingZeroes:
    def trailingZeroes(self, n: int) -> int:
        return 0 if n == 0 else n // 5 + self.trailingZeroes(n // 5)

class FindtheDuplicateNumber:
    def findDuplicate(self, nums: List[int]) -> int:
        slow = 0
        fast = 0
        while (1):
            slow = nums[nums[slow]]
            fast = nums[fast]
            if slow == fast:
                slow = 0
                while (1):
                    slow = nums[slow]
                    fast = nums[fast]
                    if slow == fast:
                        return slow

class FirstUniqueCharacterinaString:
    def firstUniqChar(self, s: str) -> int:
        _dict = dict()
        for i in s:
            _dict[i] = _dict[i]+1 if i in _dict else 1
        for i in range(len(s)):
            if _dict[s[i]] == 1:
                return i
        return -1

class FizzBuzz:
    def fizzBuzz(self, n: int) -> List[str]:
        res = []
        _dict = {3:"Fizz", 5:"Buzz"}
        for i in range(1, n+1):
            curr_string = ""
            for key in _dict.keys():
                if i%key == 0:
                    curr_string += _dict[key]
            if not curr_string:
                curr_string = str(i)
            res.append(curr_string)
        return res

class FourSumII:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        _dict = dict()
        count = 0
        for a in A:
            for b in B:
                _dict[a+b]=_dict[a+b]+1 if a+b in _dict else 1
        for c in C:
            for d in D:
                if -(c+d) in _dict:
                    count += _dict[-(c+d)]
        return count

class GameofLife:
    def gameOfLife(self, board: List[List[int]]) -> None:
        board_copy = [row[:] for row in board]
        for row in range(len(board)):
            for col in range(len(board[0])):
                self.isAlive(row, col, board, board_copy)
        for row in range(len(board)):
            for col in range(len(board[0])):
                board[row][col] = board_copy[row][col]
    
    def isAlive(self, row, col, board, board_copy):
        alive_neighbors = self.countAlive(self, row, col, board)
        if not board[row][col] and alive_neighbors == 3:
            board_copy[row][col] = 1
        elif board[row][col] and alive_neighbors != 2 and alive_neighbors != 3:
            board_copy[row][col] = 0
    
    def countAlive(self, row, col, board):
        count = 0
        if row-1 >= 0 and board[row-1][col]:
            count += 1
        if col-1 >= 0 and board[row][col-1]:
            count += 1
        if row+1 < len(board) and board[row+1][col]:
            count += 1
        if col+1 < len(board[0]) and board[row][col+1]:
            count += 1
        if row-1 >= 0 and col-1 >= 0 and board[row-1][col-1]:
            count += 1
        if row-1 >= 0 and col+1 < len(board[0]) and board[row-1][col+1]:
            count += 1 
        if row+1 < len(board) and col-1 >= 0 and board[row+1][col-1]:
            count += 1
        if row+1 < len(board) and col+1 < len(board[0]) and board[row+1][col+1]:
            count += 1 
        return count

class GenerateParenthesis:
    def generateParenthesis(self, n: int) -> List[str]:
        if not n:
            return []
        res = []
        self.dfs(res, n, n)
        return res

    def dfs(self, res, left, right, solution=""):
        if left > right:
            return
        if not left and not right:
            res.append(solution)
            return
        if left:
            self.dfs(res, left-1, right, solution+"(")
        if right:
            self.dfs(res, left, right-1, solution+")")

class GroupAnagrams:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = []
        _dict = dict()
        for word in strs:
            sort = ''.join(sorted(word))
            if sort in _dict:
                _dict[sort].append(word)
            else:
                _dict[sort] = [word]
        for val in _dict.values():
            res.append(val)
        return val

class HappyNumber:
    def isHappy(self, n: int) -> bool:
        _set = set()
        while (1):
            _sum = 0
            for i in str(n):
                _sum += int(i)**2
            if _sum == 1:
                return True
            elif _sum in _set:
                return False
            _set.add(_sum)
            n = _sum

class HouseRobber:
    def rob(self, nums: List[int]) -> int:
        current = 0
        previous = 0
        for i in nums:
            temp = previous
            previous = current
            current = max(current, temp+i)
        return current

class ImplementstrStr:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle in haystack:
            for i in range(len(haystack)-len(needle)+1):
                if haystack[i:i+len(needle)] == needle:
                    return i
        return -1

class IncreasingTripletSubsequence:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = second = float('inf')
        for i in nums:
            if i <= first:
                first = i
            elif i <= second:
                second = i
            else:
                return True
        return False

class IntersectionofTwoArraysII:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = []
        _dict = dict()
        short = nums1 if len(nums1) < len(nums2) else nums2
        long = nums2 if short == nums1 else nums1
        for i in short:
            _dict[i] = _dict[i]+1 if i in _dict else 1
        for i in long:
            if i in _dict and _dict[i] > 0:
                res.append(i)
                _dict[i] -= 1
        return res

class IntersectionofTwoLinkedLists:
    def getIntersectionNode(self, headA, headB):
        a = headA
        b = headB
        while a != b:
            if not a:
                a = headB
            else:
                a = a.next
            if not b:
                b = headA
            else:
                b = b.next
        return a

class JumpGame:
    def canJump(self, nums: List[int]) -> bool:
        lastPos = len(nums)-1
        for i in range(len(nums)-2, -1, -1):
            if nums[i]+i >= lastPos:
                lastPos = i
        return lastPos == 0

import heapq
class KthLargestElementinanArray:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return sorted(nums)[-k]

    def findKthLargest(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums)
        for i in range(len(nums)-k+1):
            k = heapq.heappop(nums)
        return k

class KthSmallestElementinaBST:
    def kthSmallest(self, root, k):
        stack = []
        while (1):
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right
    
class KthSmallestElementinaSortedMatrix:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        res = []
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                res.append(matrix[row][col])
        heapq.heapify(res)
        for i in range(k):
            k = heapq.heappop(res)
        return k

class LetterCombinationsofaPhoneNumber:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        _dict = {
            2: "abc",
            3: "def",
            4: "ghi",
            5: "jkl",
            6: "mno",
            7: "pqrs",
            8: "tuv",
            9: "wxyz"
        }

        combinations = [""]
        for digit in digits:
            new_combinations = []
            for combination in combinations:
                for letter in _dict[int(digit)]:
                    new_combinations.append(combination+letter)
            combinations = new_combinations
        return combinations
    
class LinkedListCycle:
    def hasCycle(self, head):
        if not head:
            return False
        slow = head
        fast = head.next
        while slow != fast:
            if not slow or not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        return True

class LongestCommonPrefix:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ""
        for i in zip(*strs):
            if len(set(i)) != 1:
                return res
            res += i[0]
        return res

class LongestConsecutiveSequence:
    def longestConsecutive(self, nums: List[int]) -> int:
        _set = set(nums)
        max_streak = 0
        for i in _set:
            if i-1 not in _set:
                curr_num = i
                curr_streak = 1
                while curr_num+1 in _set:
                    curr_num += 1
                    curr_streak += 1
                max_streak = max(curr_streak, max_streak)
        return max_streak

class LongestIncreasingSubsequence:
    def lengthOfLIS(self, nums: List[int]) -> int:
        sub = []
        for i in nums:
            pos = 0
            curr_len = len(sub)
            while pos <= curr_len:
                if pos == curr_len:
                    sub.append(i)
                    break
                elif i <= sub[pos]:
                    sub[pos] = i
                    break
                else:
                    pos += 1
        return len(sub)

class MajorityElement:
    def majorityElement(self, nums: List[int]) -> int:
        return sorted(nums)[len(nums)//2]

class MaximumDepthofBinaryTree:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        return 1+max(self.maxDepth(root.left), self.maxDepth(root.right))

class MaximumProductSubarray:
    def maxProduct(self, nums: List[int]) -> int:
        maxProduct = -float('inf')
        currMax = 1
        currMin = 1
        for i in nums:
            if i < 0:
                currMax, currMin = currMin, currMax
            currMax = max(i, currMax*i)
            currMin = min(i, currMin*i)
            if currMax > maxProduct:
                maxProduct = currMax
        return maxProduct

class MaximumSubarray:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSum = -float('inf')
        currMax = 0
        for i in nums:
            currMax = max(i, currMax+i)
            if currMax > maxSum:
                maxSum = currMax
        return maxSum

class MergeIntervals:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        res = []
        for interval in intervals:
            if not res or res[-1][1] < interval[0]:
                res.append(interval)
            else:
                res[-1][1] = max(res[-1][1], interval[1])
        return res

class MergeSortedArray:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        while m > 0 and n > 0:
            if nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]

class MergeTwoSortedLists:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = curr = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        curr.next = l1 or l2
        return dummy.next

class MissingNumber:
    def missingNumber(self, nums: List[int]) -> int:
        actual_sum = sum(nums)
        expected_sum = 0
        for i in range(len(nums)+1):
            expected_sum += 1
        return expected_sum - actual_sum

class MoveZeroes:
    def moveZeroes(self, nums: List[int]) -> None:
        last_zero = 0
        for i in range(len(nums)):
            if nums[i]:
                nums[last_zero], nums[i] = nums[i], nums[last_zero]
                last_zero += 1

class NumberofIslands:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        if not grid:
            return count
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == '1':
                    count += 1
                    self.isIsland(row, col, grid)
        return count
    
    def isIsland(self, row, col, grid):
        if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]) or grid[row][col] != '1':
            return
        grid[row][col] = '0'
        self.isIsland(self, row-1, col, grid)
        self.isIsland(self, row+1, col, grid)
        self.isIsland(self, row, col-1, grid)
        self.isIsland(self, row, col+1, grid)
    
class PalindromeLinkedList:
    def isPalindrome(self, head: ListNode) -> bool:
        fast = half = head
        # find half way point
        while fast and fast.next:
            half = half.next
            fast = fast.next.next
        # reverse half
        prev = None
        while half:
            curr = half
            half = half.next
            curr.next = prev
            prev = curr
        # compare half to head
        while half:
            if half.val != head.val:
                return False
            half = half.next
            head = head.next
        return True

class PascalsTriangle:
    def generate(self, numRows: int) -> List[List[int]]:
        triangle = [[1 for _ in range(i)] for i in range(1, numRows+1)]
        for row in range(2, numRows):
            for col in range(1, row):
                triangle[row][col] = triangle[row-1][col-1] + triangle[row-1][col]
        return triangle

class Permutations:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        self.dfs(nums, res)
        return res
    
    def dfs(self, nums, res, path=[]):
        if not nums:
            res.append(path)
            return
        for i in range(len(nums)):
            self.dfs(nums[:i]+nums[i+1:], res, path+[nums[i]])

class PlusOne:
    # Type and String Manipulation
    def plusOne(self, digits: List[int]) -> List[int]:
        return list(str(int(''.join(map(str, digits)))+1))
    
    # Algorithimically
    def plusOne(self, digits: List[int]) -> List[int]: 
        digits[-1] += 1
        for i in range(len(digits)-1,0,-1):
            if digits[i] != 10:
                break
            digits[i] = 0
            digits[i-1] += 1
        if digits[0] == 10:
            digits[0] = 0
            digits = [1] + digits
        return digits

class PowerofThree:
    def isPowerOfThree(self, n: int) -> bool:
        if n < 1:
            return False
        elif n == 1:
            return True
        while n >= 1:
            if n == 1:
                return True
            n /= 3
        return False

class ProductofArrayExceptSelf:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        prod = 1
        res = [1 for _ in nums]
        for i in range(1, len(nums)):
            res[i] = res[i-1] * nums[i-1]
        for i in range(len(nums)-2, -1, -1):
            prod *= nums[i+1]
            res[i] *= prod
        return res

class RemoveDuplicatesfromSortedArray:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        j = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[j]:
                j += 1
                nums[j] = nums[i]
        return j+1

class ReverseInteger:
    def reverse(self, x: int) -> int:
        if str(x)[0] == "-":
            if -int(str(x)[1:][::-1]) >= -2**31:
                return -int(str(x)[1:][::-1])
            return 0
        else:
            if int(str(x)[::-1]) <= 2**31-1:
                return int(str(x)[::-1])
            return 0

class ReverseLinkedList:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        while head:
            curr = head
            head = head.next
            curr.next = prev
            prev = curr
        return prev

class ReverseString:
    def reverseString(self, s: List[str]) -> None:
        start = 0
        end = len(s)-1
        while start < end:
            s[start], s[end] = s[start], s[end]
            start += 1
            end -= 1
        
class RomantoInteger:
    def romanToInt(self, s: str) -> int:
        _dict = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        prev = 0
        _sum = 0
        for i in s[::-1]:
            curr = _dict[i]
            if curr < prev:
                _sum -= curr
            else:
                _sum += curr
            prev = curr
        return _sum

class RotateArray:
    def rotate(self, nums: List[int], k: int) -> None:
        k %= len(nums)
        nums.reverse()
        self.reverse(0, k-1, nums)
        self.reverse(k, len(nums)-1, nums)

    def reverse(self, start, end, nums):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1

class RotateImage:
    def rotate(self, matrix: List[List[int]]) -> None:
        matrix.reverse()
        for row in range(len(matrix)):
            for col in range(row):
                matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]

class Searcha2DMatrixII:
    def searchMatrix(self, matrix, target):
        if not matrix:
            return False
        row = 0
        col = len(matrix[0])-1
        while row < len(matrix) and col >= 0:
            if matrix[row][col] < target:
                row += 1
            elif matrix[row][col] > target:
                col -= 1
            else:
                return True
        return False

class SetMatrixZeroes:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        set_col = False
        for row in range(len(matrix)):
            if not matrix[row][0]:
                set_col = True
            for col in range(1, len(matrix[0])):
                if not matrix[row][col]:
                    matrix[row][0] = matrix[0][col] = 0

        for row in range(1, len(matrix)):
            for col in range(1, len(matrix[0])):
                if not matrix[row][0] or not matrix[0][col]:
                    matrix[row][col] = 0

        if not matrix[0][0]:
            for col in range(len(matrix[0])):
                matrix[0][col] = 0

        if set_col:
            for row in range(len(matrix)):
                matrix[row][0] = 0

class SingleNumber:
    # O(n) time, O(n) memory
    def singleNumber(self, nums: List[int]) -> int:
        _dict = dict()
        for i in nums:
            _dict[i] = _dict[i]+1 if i in _dict else 1
        for key, value in _dict.items():
            if value == 1:
                return key

    # O(nlogn) time, O(1) memory
    def singleNumber(self, nums: List[int]) -> int:
        nums.sort()
        first_occurrence = True
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                if not first_occurrence:
                    first_occurrence = True
                else:
                    return nums[i-1]
            else:
                first_occurrence = False
        if first_occurrence:
            return nums[-1]

class SortColors:
    # O(n) memory
    def sortColors(self, nums: List[int]) -> None:
        _dict = {0:0, 1:0, 2:0}
        for i in nums:
            _dict[i] += 1
        index = 0
        for i in range(_dict[0]):
            nums[index] = 0
            index += 1
        for i in range(_dict[1]):
            nums[index] = 1
            index += 1
        for i in range(_dict[2]):
            nums[index] = 2
            index += 1
            
    # O(1) memory
    def sortColors(self, nums: List[int]) -> None:
        i = j = 0
        for k in range(len(nums)):
            curr = nums[k]
            nums[k] = 2
            if curr < 2:
                nums[i] = 1
                i += 1
            if not curr:
                nums[j] = 0
                j += 1

class SortList:
    def sortList(self, head: ListNode) -> ListNode:
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        self.mergeSort(arr)
        head = dummy = ListNode(0)
        for i in arr:
            dummy.next = ListNode(i)
            dummy = dummy.next
        return head.next

    def mergeSort(self, arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]
            self.mergeSort(left)
            self.mergeSort(right)
            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1
            
            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1

            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1

class Sqrt(x):
    def mySqrt(self, x: int) -> int:
        lo, hi = 0, x
        while lo < hi:
            mid = lo + (hi - lo + 1) // 2
            if mid**2 <= hi:
                lo = mid
            else:
                hi = mid - 1
        return lo 

class Subsets: 
    def subsets(self, nums):
        res = []
        self.dfs(res, nums)
        return res
    
    def dfs(self, res, nums, index=0, path=[]):
        res.append(path)
        for i in range(index, len(nums)):
            self.dfs(res, nums, i+1, path+[nums[i]])

class SymmetricTree:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return self.isMirror(root.left, root.right)
    
    def isMirror(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return left.val == right.val and self.isMirror(left.right, right.left) and self.isMirror(left.left, right.right)

class TopKFrequentElements:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        _dict = dict()
        for i in nums:
            _dict[i] = _dict[i]+1 if i in _dict else 1
        return sorted(_dict, key=_dict.get, reverse=True)[:k]

class TwoSum:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = dict()
        for i in range(len(nums)):
            if target-nums[i] in _dict:
                return [_dict[target-nums[i]], i]
            _dict[nums[i]] = i

class UniquePaths:
    def uniquePaths(self, m: int, n: int) -> int:
        matrix = [[1 for _ in range(m)] for _ in range(n)]
        for row in range(1, n):
            for col in range(1, m):
                matrix[row][col] = matrix[row-1][col] + matrix[row][col-1]
        return matrix[-1][-1]

class ValidAnagram:
    def isAnagram(self, s: str, t: str) -> bool:
        _dict = dict()
        if len(s) != len(t):
            return False
        for i in s:
            _dict[i] = _dict[i]+1 if i in _dict else 1
        for i in t:
            if i in _dict and _dict[i] > 0:
                _dict[i] -= 1
            else:
                return False
        return True

class ValidPalindrome:
    def isPalindrome(self, s: str) -> bool:
        list_of_letters = [letter.lower() for letter in s if letter.isalnum()]
        return list_of_letters == list_of_letters[::-1]

class ValidParentheses:
    def isValid(self, s: str) -> bool:
        stack = []
        _dict = {'}': '{', ']': '[', ')': '('}
        for i in s:
            if i in _dict:
                top = stack.pop() if stack else None
                if top != _dict[i]:
                    return False
            else:
                stack.append(i)
        return not stack

class ValidSudoku:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        return self.isSquareValid(board) and self.isRowValid(board) and self.isColValid(board)

    def isSquareValid(self, board):
        for row in (0,3,6):
            for col in (0,3,6):
                square = []
                for x in range(row, row+3):
                    for y in range(col, col+3):
                        square.append(board[x][y])
                if not self.isUnitValid(square):
                    return False
        return True

    def isRowValid(self, board):
        for row in board:
            if not self.isUnitValid(row):
                return False
        return True

    def isColValid(self, board):
        for col in zip(*board):
            if not self.isUnitValid(col):
                return False
        return True

    def isUnitValid(self, square):
        square = [i for i in square if i != "."]
        return len(set(square)) == len(square)

class ValidateBinarySearchTree:
    def isValidBST(self, root: TreeNode) -> bool:
        stack = []
        inorder = float('-inf')
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= inorder:
                return False
            inorder = root.val
            root = root.right
        return True

class WordBreak:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False for _ in range(len(s)+1)]
        dp[0] = True
        for i in range(len(s)):
            if dp[i]:
                for j in range(i, len(s)):
                    if s[i:j+1] in wordDict:
                        dp[j+1] = True
        return dp[-1]

class WordSearch:
    def exist(self, board: List[List[str]], word: str) -> bool:
        for row in range(len(board)):
            for col in range(len(board[0])):
                if self.dfs(row, col, board, word):
                    return True
        return False

    def dfs(self, row, col, board, word):
        if not word:
            return True
        if row < 0 or col < 0 or row == len(board) or col == len(board[0]) or word[0] != board[row][col]:
            return False
        temp = board[row][col]
        board[row][col] = None
        result = self.dfs(row-1, col, board, word[1:]) or self.dfs(row+1, col, board, word[1:]) or self.dfs(row, col-1, board, word[1:]) or self.dfs(row, col+1, board, word[1:])
        board[row][col] = temp
        return result