package main

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

func findSubstring(s string, words []string) []int {
	ans := make([]int, 0)
	n, cnt := len(s), len(words)
	if n <= 0 || cnt <= 0 {
		return ans
	}

	dict := make(map[string]int)
	for i := 0; i < cnt; i++ {
		dict[words[i]]++
	}

	wl := len(words[0])
	for i := 0; i < wl; i++ {
		fmt.Println("i:", i)
		left, count := i, 0
		tdict := make(map[string]int)
		for j := i; j <= n-wl; j += wl {
			str := s[j : j+wl]
			fmt.Println(tdict)
			_, present := dict[str]
			if present {
				tdict[str]++
				if tdict[str] <= dict[str] {
					count++
				} else {
					for tdict[str] > dict[str] {
						str1 := s[left : left+wl]
						if tdict[str1] <= dict[str1] {
							count--
						}
						tdict[str1]--
						left += wl
						fmt.Println("shift", tdict)
					}
				}
				if count == cnt {
					ans = append(ans, left)
					tdict[s[left:left+wl]]--
					count--
					left += wl
				}
			} else {
				tdict = make(map[string]int)
				count = 0
				left = j + wl
			}

		}
	}

	return ans
}

func nextPermutation(nums []int) {
	if len(nums) <= 1 {
		return
	}

	var i int
	for i = len(nums) - 2; i >= 0; i-- {
		if nums[i] < nums[i+1] {
			break
		}
	}

	var j int
	if i >= 0 {
		for j = len(nums) - 1; j > i; j-- {
			if nums[j] > nums[i] {
				break
			}
		}
		nums[i], nums[j] = nums[j], nums[i]
	}

	//reverse
	for m, n := i+1, len(nums)-1; m < n; m, n = m+1, n-1 {
		nums[m], nums[n] = nums[n], nums[m]
	}
}

type Stack []rune

func (s Stack) pop() (Stack, rune, error) {
	var r rune
	if len(s) <= 0 {
		return s, r, errors.New("empty")
	}
	r = s[len(s)-1]
	s = s[:len(s)-1]
	return s, r, nil
}

func (s Stack) push(e rune) Stack {
	return append(s, e)
}

func valid(s string) bool {
	var sta Stack
	for _, el := range s {
		if el == '(' {
			sta = sta.push(el)
		} else if el == ')' {
			var la rune
			var err error
			sta, la, err = sta.pop()
			if err != nil {
				return false
			}
			if la != '(' {
				return false
			}
		}
	}
	if len(sta) != 0 {
		return false
	}
	return true
}

func longestValidParentheses1(s string) int {
	max := 0
	for i := 0; i < len(s); i++ {
		for j := i + 1; j <= len(s); j++ {
			if valid(s[i:j]) {
				if j-i > max {
					max = j - i
				}
			}
		}
	}

	return max
}

func longestValidParentheses(s string) int {
	dp := make([]int, len(s))
	max := 0
	for i := 1; i < len(s); i++ {
		if s[i] == ')' {
			if s[i-1] == '(' {
				if i-2 >= 0 {
					dp[i] = dp[i-2] + 2
				} else {
					dp[i] = 2
				}
			} else if i-dp[i-1] > 0 && s[i-dp[i-1]-1] == '(' {
				if i-dp[i-1]-2 >= 0 {
					dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
				} else {
					dp[i] = dp[i-1] + 2
				}
			}
			if dp[i] > max {
				max = dp[i]
			}
		}
	}

	return max
}

func extremeInert(nums []int, target int, left bool) int {
	lo := 0
	hi := len(nums)
	for lo < hi {
		mid := (lo + hi) / 2
		if nums[mid] > target || (left && target == nums[mid]) {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	fmt.Println(lo, hi)
	return lo
}

func searchRange(nums []int, target int) []int {
	ret := []int{-1, -1}
	left := extremeInert(nums, target, true)

	if left == len(nums) || nums[left] != target {
		return ret
	}

	ret[0] = left
	ret[1] = extremeInert(nums, target, false) - 1
	return ret
}

func newgrid() [][]bool {
	g := make([][]bool, 9)
	for i := 0; i < 9; i++ {
		g[i] = make([]bool, 9)
	}
	return g
}

func isValidSudoku(board [][]byte) bool {
	use1 := newgrid()
	use2 := newgrid()
	use3 := newgrid()

	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] == '.' {
				continue
			}
			num := board[i][j] - '0' - 1
			k := i/3*3 + j/3
			if use1[i][num] || use2[j][num] || use3[k][num] {
				return false
			}
			use1[i][num], use2[j][num], use3[k][num] = true, true, true
		}
	}
	return true
}

func validSk(board [][]byte, row, col int, c byte) bool {
	for i := 0; i < 9; i++ {
		if board[i][col] != '.' && board[i][col] == c {
			return false
		}
		if board[row][i] != '.' && board[row][i] == c {
			return false
		}
		ch := board[row/3*3+i/3][col/3*3+i%3]
		if ch != '.' && ch == c {
			return false
		}
	}
	return true
}

func solve(board [][]byte) bool {
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] == '.' {
				for c := '1'; c <= '9'; c++ {
					if validSk(board, i, j, byte(c)) {
						board[i][j] = byte(c)

						if solve(board) {
							return true
						} else {
							board[i][j] = '.'
						}
					}
				}

				return false
			}
		}
	}

	return true

}

func solveSudoku(board [][]byte) {
	if len(board) == 0 {
		return
	}
	solve(board)
}

/*func combinationSum(candidates []int, target int) [][]int {
	ans := make([][]int,0)
	if len(candidates) == 1 && candidates[0] == target{
		ans = append(ans,candidates)
		return ans
	}
	if len(candidates) == 0{
		return ans
	}
	for i:=0; i<len(candidates);i++{
		new_target := target-candidates[i]
		res := combinationSum(candidates[i+1:],new_target)
		for _,r := range res{

		}
	}
}*/

func sort(a []int) {
	for i := 0; i < len(a); i++ {
		for j := i + 1; j < len(a); j++ {
			if a[i] > a[j] {
				a[i], a[j] = a[j], a[i]
			}
		}
	}
}

func backtrack(can []int, ret *[][]int, tmp []int, remain, start int) {
	if remain < 0 {
		return
	} else if remain == 0 {
		newtmp := make([]int, 0)
		newtmp = append(newtmp, tmp...)
		*ret = append(*ret, newtmp)
	} else {
		for i := start; i < len(can); i++ {
			if i > start && can[i] == can[i-1] {
				continue
			}
			tmp = append(tmp, can[i])
			backtrack(can, ret, tmp, remain-can[i], i+1)
			tmp = tmp[:len(tmp)-1]
		}
	}
}

func combinationSum2(candidates []int, target int) [][]int {
	ret := make([][]int, 0)
	tmp := make([]int, 0)

	sort(candidates)
	backtrack(candidates, &ret, tmp, target, 0)

	return ret
}

/*func firstMissingPositive(nums []int) int {
	for i:=0; i < len(nums); i++{
		for nums[i] > 0 && nums[i] <= len(nums) && nums[nums[i] - 1] != nums[i] {
			nums[i],nums[nums[i]-1] = nums[nums[i]-1],nums[i]
		}
	}

	for i:=0; i < len(nums); i++{
		if nums[i] != i+1{
			return i+1
		}
	}
	return len(nums)+1

}*/

func firstMissingPositive(nums []int) int {
	pre, last := 0, len(nums)-1
	for pre < last {
		if nums[pre] <= 0 {
			pre++
		} else if nums[last] > 0 {
			last--
		} else { //nums[pre] > 0 && nums[last] <= 0
			nums[pre], nums[last] = nums[last], nums[pre]
			pre++
			last--
		}
	}
	fmt.Println(nums)
	//find first positive number
	var i int
	for i = 0; i < len(nums); i++ {
		if nums[i] > 0 {
			break
		}
	}

	for j := i; j < len(nums); j++ {
		if nums[j] != j-i+1 {
			return j - i + 1
		}
	}
	return len(nums) + 1

}

func leftMax(n []int, pos int) int {
	max := 0
	for i := 0; i < pos; i++ {
		if n[i] > max {
			max = n[i]
		}
	}
	return max
}

func rightMax(n []int, pos int) int {
	max := 0
	for i := pos + 1; i < len(n); i++ {
		if n[i] > max {
			max = n[i]
		}
	}
	return max
}

func min(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

/*func trap(height []int) int {
	full := 0
	for i:=0;i<len(height); i++{
		left := leftMax(height,i)
		right := rightMax(height,i)
		lr_min := min(left,right)
		stor := lr_min - height[i]
		if stor > 0{
			full += stor
		}
	}

	return full
}*/

func max(a, b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}

func trap(height []int) int {
	size := len(height)
	left := make([]int, size)
	right := make([]int, size)

	left[0] = height[0]
	for i := 1; i < size; i++ {
		left[i] = max(height[i], left[i-1])
	}

	right[size-1] = height[size-1]
	for i := size - 2; i >= 0; i-- {
		right[i] = max(height[i], right[i+1])
	}
	full := 0
	for i := 0; i < size; i++ {
		lr_min := min(left[i], right[i])
		stor := lr_min - height[i]
		if stor > 0 {
			full += stor
		}
	}
	return full
}

func multiply(num1 string, num2 string) string {
	s1, s2 := len(num1), len(num2)
	ans := make([]int, 100000)
	var pos int
	for i := s1 - 1; i >= 0; i-- {
		//carry := 0
		c1 := int(num1[i] - '0')
		for j := s2 - 1; j >= 0; j-- {
			c2 := int(num2[j] - '0')
			pos = (s1 - i - 1) + (s2 - j - 1)
			ans[pos] = c1*c2 + ans[pos]
			for ans[pos] > 10 {
				carry := ans[pos] / 10
				ans[pos] %= 10
				ans[pos+1] += carry
				pos++
			}
		}
	}
	strans := ""
	for pos >= 0 && ans[pos] == 0 {
		pos--
	}
	for ; pos >= 0; pos-- {
		strans += strconv.Itoa(ans[pos])
	}
	if strans == "" {
		return "0"
	}
	return strans

}

/*func isMatch(s string, p string) bool {
	if len(p) == 0{
		return len(s) == 0
	}

	if p[0] == '?' || s[0] = p[0]{
		return isMatch(s[1:],p[:])
	}else if p[0] == '*'{
		return isMatch()
	}else{
		return false
	}
}*/

func isMatchRE(s string, p string) bool {
	if len(p) == 0 {
		return len(s) == 0
	}
	first_match := len(s) > 0 && (s[0] == p[0] || p[0] == '.')
	if len(p) >= 2 && p[1] == '*' {
		return isMatchRE(s, p[2:]) || (first_match && isMatchRE(s[1:], p))
	} else {
		return first_match && isMatchRE(s[1:], p[1:])
	}

}

func jump(nums []int) int {
	if len(nums) < 2 {
		return 0
	}
	level := 0
	lnums := []int{0}
	for len(lnums) != 0 {
		level++
		new_lnums := make([]int, 0)
		cur_max := 0
		for _, v := range lnums {
			num_v := nums[v]
			if v+num_v > cur_max {
				cur_max = v + num_v
				for i := 1; i <= num_v; i++ {
					new_pos := v + i
					if new_pos == len(nums)-1 {
						return level
					}
					if new_pos > len(nums)-1 {
						break
					}
					if len(new_lnums) > 0 && new_pos <= new_lnums[len(new_lnums)-1] {
						continue
					}
					new_lnums = append(new_lnums, new_pos)
				}
			}
		}
		lnums = new_lnums
	}
	return 0

}

func permute(nums []int) [][]int {
	ret := make([][]int, 0)
	//fmt.Println("passing",nums)
	//defer func(){fmt.Println("return",ret)}()
	if len(nums) == 0 {
		return ret
	}
	if len(nums) == 1 {
		ret = append(ret, nums)
		return ret
	}
	for i := 0; i < len(nums); i++ {
		//fmt.Println(i)
		//fmt.Println(nums[i])
		//fmt.Println("nums",nums)
		new_nums := make([]int, 0)
		new_nums = append(new_nums, nums[:i]...)
		new_nums = append(new_nums, nums[i+1:]...)
		tmp_ret := permute(new_nums)
		//fmt.Println("after nums",nums)
		for _, v := range tmp_ret {
			tt := []int{nums[i]}
			tt = append(tt, v...)
			ret = append(ret, tt)
		}
	}

	return ret
}

func permuteU(nums []int) [][]int {
	ret := make([][]int, 0)
	if len(nums) == 0 {
		return ret
	}
	if len(nums) == 1 {
		ret = append(ret, nums)
		return ret
	}
	for i := 0; i < len(nums); i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		new_nums := make([]int, 0)
		new_nums = append(new_nums, nums[:i]...)
		new_nums = append(new_nums, nums[i+1:]...)
		tmp_ret := permuteU(new_nums)
		for _, v := range tmp_ret {
			tt := []int{nums[i]}
			tt = append(tt, v...)
			ret = append(ret, tt)
		}
	}

	return ret
}

func permuteUnique(nums []int) [][]int {
	sort(nums)
	return permuteU(nums)
}

func flip(matrix [][]int) {
	for i := 0; i < len(matrix)/2; i++ {
		matrix[i], matrix[len(matrix)-1-i] = matrix[len(matrix)-1-i], matrix[i]
	}
}

func rotate(matrix [][]int) {
	//flip first,then
	flip(matrix)
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if i > j {
				matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
			}
		}
	}
}

func validQ(mt [][]bool, row, col int) bool {
	//row
	for i := 0; i < col; i++ {
		if mt[row][i] {
			return false
		}
	}
	//line
	for i := 0; i < row; i++ {
		if mt[i][col] {
			return false
		}
	}
	//xie
	for i, j := row-1, col-1; i >= 0 && j >= 0; i, j = i-1, j-1 {
		if mt[i][j] {
			return false
		}
	}

	for i, j := row-1, col+1; i >= 0 && j < len(mt); i, j = i-1, j+1 {
		if mt[i][j] {
			return false
		}
	}
	return true
}

func solveNq(ans *[][]string, mt [][]bool, row int) {
	if row == len(mt) {
		//valid ans, add to ans later
		tmpans := make([]string, 0)
		for i := 0; i < len(mt); i++ {
			ans_str := ""
			for j := 0; j < len(mt[i]); j++ {
				if mt[i][j] {
					ans_str += "Q"
				} else {
					ans_str += "."
				}
			}
			tmpans = append(tmpans, ans_str)
		}
		fmt.Println(tmpans)
		*ans = append(*ans, tmpans)
		return
	}
	for i := 0; i < len(mt[row]); i++ {
		mt[row][i] = true
		if validQ(mt, row, i) {
			solveNq(ans, mt, row+1)
		}
		mt[row][i] = false
	}
}

func solveNQueens(n int) [][]string {
	solu := make([][]string, 0)
	mt := make([][]bool, n)
	for i := 0; i < n; i++ {
		mt[i] = make([]bool, n)
	}
	solveNq(&solu, mt, 0)
	return solu
}

func totalNQueens(n int) int {
	return len(solveNQueens(n))
}

func spiralOrder(matrix [][]int) []int {
	dir := make([][]int, 4)
	dir[0] = []int{0, 1}
	dir[1] = []int{1, 0}
	dir[2] = []int{0, -1}
	dir[3] = []int{-1, 0}
	if len(matrix) < 1 {
		return []int{}
	}
	ans := make([]int, 0, len(matrix)*len(matrix[0]))
	visited := make([][]bool, len(matrix))
	for i := 0; i < len(visited); i++ {
		visited[i] = make([]bool, len(matrix[i]))
	}
	d := 0
	for i, j := 0, -1; ; {
		//try to add
		newi, newj := i+dir[d][0], j+dir[d][1]
		if newi < 0 || newi >= len(matrix) || newj < 0 || newj >= len(matrix[newi]) || visited[newi][newj] {
			//fail
			d = (d + 1) % 4
			continue
		}
		//success
		i, j = newi, newj
		visited[i][j] = true
		ans = append(ans, matrix[i][j])
		if len(ans) == len(matrix)*len(matrix[0]) {
			break
		}
	}
	return ans
}

/*func canJump(nums []int) bool {
	if len(nums) <= 1{
		return true
	}
	pos := make([]int,0)
	pos = append(pos,0)
	for len(pos) != 0{
		fmt.Println(pos)
		if pos[0] >= len(nums){
			return false
		}
		new_pos := make([]int,0)
		for i:=0; i<len(pos); i++{
			if pos[i] >= len(nums){
				continue
			}
			value := nums[pos[i]]
			j := 1
			for ; j <= value ; j++{
				new_p := pos[i] + j
				if new_p == len(nums) - 1{
					return true
				}
				if len(new_pos) > 0 && new_p <= new_pos[len(new_pos)-1]{
					j = new_pos[len(new_pos)-1]
					continue
				}
				//if new_p > pos[len(pos)-1]{
					new_pos = append(new_pos,new_p)
				//}
			}
		}
		pos = new_pos
	}
	return false

}*/

func canJump(nums []int) bool {
	size := len(nums) - 1
	left := size - 1
	for j := size - 1; j >= 0; j-- {
		if j+nums[j] >= left {
			left = j
		}
	}
	return left == 0
}

/**
 * Definition for an interval.
 * type Interval struct {
 *	   Start int
 *	   End   int
 * }
 */

type Interval struct {
	Start int
	End   int
}

func sortInterval(a []Interval) {
	for i := 0; i < len(a); i++ {
		for j := i + 1; j < len(a); j++ {
			if a[i].Start > a[j].Start {
				a[i], a[j] = a[j], a[i]
			}
		}
	}
}

func merge(intervals []Interval) []Interval {
	if len(intervals) <= 1 {
		return intervals
	}
	sortInterval(intervals)
	ans := make([]Interval, 0)
	left := intervals[0].Start
	right := intervals[0].End
	var i int
	for i = 1; i < len(intervals); i++ {
		//if intervals[i].Start <= intervals[i-1].End{
		if intervals[i].Start <= right {
			right = max(right, intervals[i-1].End)
			right = max(intervals[i].End, right)

			continue
		} else {
			tmp := Interval{left, right}
			ans = append(ans, tmp)
			left = intervals[i].Start
			right = max(right, intervals[i].End)
		}
	}
	tmp := Interval{left, right}
	ans = append(ans, tmp)
	return ans
}

func insert(intervals []Interval, newInterval Interval) []Interval {
	intervals = append(intervals, newInterval)
	return merge(intervals)
}

func generateMatrix(n int) [][]int {
	dir := make([][]int, 4)
	dir[0] = []int{0, 1}
	dir[1] = []int{1, 0}
	dir[2] = []int{0, -1}
	dir[3] = []int{-1, 0}
	if n < 1 {
		return [][]int{}
	}
	visited := make([][]int, n)
	for i := 0; i < len(visited); i++ {
		visited[i] = make([]int, n)
	}
	d := 0
	i, j := 0, -1
	cnt := 1
	for cnt <= n*n {
		//try to add
		newi, newj := i+dir[d][0], j+dir[d][1]
		if newi < 0 || newi >= n || newj < 0 || newj >= n || visited[newi][newj] != 0 {
			//fail
			d = (d + 1) % 4
			continue
		}
		//success
		i, j = newi, newj
		visited[i][j] = cnt
		cnt++
	}
	return visited

}

func getjie(n int) int {
	ans := 1
	for i := 1; i <= n; i++ {
		ans *= i
	}
	return ans
}

func getp(nums []string, n int, k int) string {
	fmt.Println(nums)
	ans := ""
	if len(nums) == 1 {
		return nums[0]
	}
	jie := getjie(n - 1)
	index := k / jie
	index_v := nums[index]
	nums = append(nums[:index], nums[index+1:]...)
	ans = index_v + getp(nums, n-1, k-index*jie)
	return ans

}

func getPermutation(n int, k int) string {
	nums := make([]string, n)
	for i := 1; i <= n; i++ {
		nums[i-1] = string('0' + i)
	}

	return getp(nums, n, k-1)
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k == 0 {
		return head
	}
	length := 1
	//get length
	l := head
	for l.Next != nil {
		l = l.Next
		length++
	}

	k = k % length
	if head == nil || head.Next == nil || k == 0 {
		return head
	}
	fast := head
	slow := head
	for i := 0; i < k; i++ {
		if fast == nil || fast.Next == nil {
			return head
		}
		fast = fast.Next
	}
	for fast.Next != nil {
		fast, slow = fast.Next, slow.Next
	}
	tmphead := slow.Next
	fast.Next = head
	slow.Next = nil
	return tmphead

}

func up(grid [][]int, orow, ocol, m, n int) int {
	if grid[orow][ocol] != 0 {
		return grid[orow][ocol]
	}
	row, col := orow, ocol
	if row == m-1 && col == n-1 {
		fmt.Println("return 1")
		return 1
	}
	//down
	ans := 0
	for row = row + 1; row < m; row++ {
		ans += up(grid, row, ocol, m, n)
		break
	}
	//right
	for col = col + 1; col < n; col++ {
		ans += up(grid, orow, col, m, n)
		break
	}
	grid[orow][ocol] = ans
	return ans

}

func uniquePaths(m int, n int) int {
	grid := make([][]int, m)
	for i := 0; i < m; i++ {
		grid[i] = make([]int, n)
	}
	return up(grid, 0, 0, m, n)
}

func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	if len(obstacleGrid) == 0 {
		return 0
	}
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	grid := make([][]int, m)
	for i := 0; i < m; i++ {
		grid[i] = make([]int, n)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if obstacleGrid[i][j] == 1 {
				grid[i][j] = 0
			} else if i == 0 && j == 0 {
				grid[i][j] = 1
			} else if i == 0 && j > 0 {
				grid[i][j] = grid[i][j-1]
			} else if i > 0 && j == 0 {
				grid[i][j] = grid[i-1][j]
			} else {
				grid[i][j] = grid[i-1][j] + grid[i][j-1]
			}

		}
	}
	fmt.Println(grid)
	return grid[m-1][n-1]
}

func minPathSum(grid [][]int) int {
	if len(grid) == 0 {
		return 0
	}
	m, n := len(grid), len(grid[0])
	sumgrid := make([][]int, m)
	for i := 0; i < m; i++ {
		sumgrid[i] = make([]int, n)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 && j == 0 {
				sumgrid[i][j] = grid[i][j]
			} else if i == 0 && j > 0 {
				sumgrid[i][j] = sumgrid[i][j-1] + grid[i][j]
			} else if i > 0 && j == 0 {
				sumgrid[i][j] = sumgrid[i-1][j] + grid[i][j]
			} else {
				sumgrid[i][j] = min(sumgrid[i-1][j], sumgrid[i][j-1]) + grid[i][j]
			}

		}
	}
	return sumgrid[m-1][n-1]
}

func jlen(a []string, sp string) int {
	s := strings.Join(a, sp)
	return len(s)
}

func emptys(n int) string {
	s := ""
	for i := 0; i < n; i++ {
		s += " "
	}
	return s
}

func fullJustify(words []string, maxWidth int) []string {
	if len(words) <= 0 || maxWidth == 0 {
		return words
	}
	ans := [][]string{[]string{}}
	row := 0
	for i := 0; i < len(words); {
		ll := 0
		if len(ans[row]) == 0 {
			ll = len(words[i])
		} else {
			ll = jlen(ans[row], " ") + len(words[i]) + 1
		}
		if ll > maxWidth {
			row++
			ans = append(ans, make([]string, 0))
		} else {
			ans[row] = append(ans[row], words[i])
			i++
		}
	}
	//ans[row] = append(ans[row],"")
	str_ans := make([]string, len(ans))
	for i := 0; i < len(ans); i++ {
		s := ""
		if i == len(ans)-1 { //last one
			s = strings.Join(ans[i], " ")
			s += emptys(maxWidth - len(s))
		} else if len(ans[i]) == 1 {
			s += (ans[i][0] + emptys(maxWidth-len(ans[i][0])))
		} else {
			base := (maxWidth - jlen(ans[i], "")) / (len(ans[i]) - 1)
			extra := (maxWidth - jlen(ans[i], "")) % (len(ans[i]) - 1)
			for j := 0; j < len(ans[i]); j++ {
				s += ans[i][j]
				if j == len(ans[i])-1 {
					continue
				}
				s += emptys(base)
				if extra > 0 {
					s += " "
					extra--
				}
			}
		}
		str_ans[i] = s

	}
	return str_ans
}

type MyS struct {
	l      []int
	length int
}

func (s *MyS) pop() (int, error) {
	if s.length == 0 {
		return 0, errors.New("empty")
	}
	last := s.l[s.length-1]
	s.l = s.l[:s.length-1]
	s.length--
	return last, nil
}

func (s *MyS) push(v int) {
	s.l = append(s.l, v)
	s.length++
}

func (s *MyS) empty() bool {
	return s.length == 0
}

func (s *MyS) back() int {
	return s.l[s.length-1]
}

/*func simplifyPath(path string) string {
	spath := strings.Split(path,"/")
	s := MyS{}
	for i:=0; i < len(spath); i++{
		vl := spath[i]
		if vl == ".."{
			s.pop()
		}else if vl == "." || vl == ""{

		}else{
			s.push(vl)
		}
	}
	return "/" + strings.Join(s.l,"/")
}*/

func minDistance(word1 string, word2 string) int {
	m := len(word1)
	n := len(word2)
	if m == 0 {
		return n
	}
	if n == 0 {
		return m
	}

	grid := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		grid[i] = make([]int, n+1)
	}

	for i := 0; i <= m; i++ {
		for j := 0; j <= n; j++ {
			if j == 0 {
				grid[i][j] = i
			} else if i == 0 {
				grid[i][j] = j
			} else {
				if word1[i-1] == word2[j-1] {
					grid[i][j] = grid[i-1][j-1]
				} else {
					minv := min(grid[i-1][j-1], grid[i-1][j])
					minv = min(minv, grid[i][j-1])
					grid[i][j] = minv + 1
				}
			}
		}
	}

	return grid[m][n]
}

func setZeroes(matrix [][]int) {
	m, n := len(matrix), len(matrix[0])
	col0 := false
	row0 := false

	for j := 0; j < n; j++ {
		if matrix[0][j] == 0 {
			row0 = true
			break
		}
	}
	for i := 0; i < m; i++ {
		if matrix[i][0] == 0 {
			col0 = true
			break
		}
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if matrix[i][j] == 0 {
				matrix[i][0] = 0
				matrix[0][j] = 0
			}
		}
	}
	for j := 0; j < n; j++ {
		if matrix[0][j] == 0 {
			for i := 0; i < m; i++ {
				matrix[i][j] = 0
			}
		}
	}
	for i := 0; i < m; i++ {
		if matrix[i][0] == 0 {
			for j := 0; j < n; j++ {
				matrix[i][j] = 0
			}
		}
	}
	if col0 {
		for i := 0; i < m; i++ {
			matrix[i][0] = 0
		}
	}
	if row0 {
		for j := 0; j < n; j++ {
			matrix[0][j] = 0
		}
	}

}

func searchMatrix(matrix [][]int, target int) bool {
	//search for row
	m := len(matrix)
	if m == 0 || len(matrix[0]) == 0 {
		return false
	}
	i := 0
	for ; i < m; i++ {
		if matrix[i][0] == target {
			return true
		} else if matrix[i][0] > target {
			break
		}
	}
	if i-1 >= m || i-1 < 0 {
		return false
	}
	//search in row,binary search
	target_row := matrix[i-1]
	begin, end := 0, len(target_row)
	for begin < end {
		mid := (begin + end) / 2
		if target_row[mid] == target {
			return true
		} else if target_row[mid] > target {
			end = mid
		} else {
			begin = mid + 1
		}
	}
	return false
}

func pow(x, y int) int {
	ans := 1
	for i := 0; i < y; i++ {
		ans *= x
	}
	return ans
}

func titleToNumber(s string) int {
	ans := 0
	for i := len(s) - 1; i >= 0; i-- {
		ans += int(s[i]-'A'+1) * pow(26, len(s)-i-1)
	}
	return ans
}

func nextGreatestLetter(letters []byte, target byte) byte {
	i := 0
	for ; i < len(letters); i++ {
		if letters[i] > target {
			return letters[i]
		}
	}
	return letters[0]
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseBetween(head *ListNode, m int, n int) *ListNode {
	dummy := &ListNode{}
	dummy.Next = head
	//pre start then
	pre := dummy
	for i := 0; i < m-1; i++ {
		pre = pre.Next
	}
	start := pre.Next
	then := start.Next
	for i := 0; i < n-m; i++ {
		start.Next = then.Next
		then.Next = pre.Next
		pre.Next = then
		then = start.Next
	}
	return dummy.Next

}

func minCostClimbingStairs(cost []int) int {
	l := len(cost)
	dp := make([]int, l)
	dp[l-1] = 0
	dp[l-2] = cost[l-2]
	for i := l - 3; i >= 0; i-- {
		dp[i] = cost[i] + min(dp[i+1], dp[i+2])
	}
	return dp[0]
}

func dpnum(start, end int, dp [][]int) int {
	if start > end {
		return 1
	}
	if dp[start][end] != 0 {
		return dp[start][end]
	}
	if start >= end {
		dp[start][end] = 1
		return 1
	}
	ans := 0
	for i := start; i <= end; i++ {
		ans += (dpnum(start, i-1, dp) * dpnum(i+1, end, dp))
	}
	dp[start][end] = ans
	return ans
}

func numTrees(n int) int {
	dp := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = make([]int, n+1)
	}
	ans := dpnum(1, n, dp)
	return ans
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func dpgene(start, end int) []*TreeNode {
	ans := make([]*TreeNode, 0)
	if start > end {
		ans = append(ans, nil)
		return ans
	}
	if start == end {
		node := &TreeNode{}
		node.Val = start
		ans = append(ans, node)
		return ans
	}
	for i := start; i <= end; i++ {
		lefts := dpgene(start, i-1)
		rights := dpgene(i+1, end)

		for _, l := range lefts {
			for _, r := range rights {
				node := &TreeNode{}
				node.Val = i
				node.Left = l
				node.Right = r
				ans = append(ans, node)
			}
		}
	}
	return ans
}

func generateTrees(n int) []*TreeNode {
	ans := make([]*TreeNode, 0)
	for _, v := range dpgene(1, n) {
		if v != nil {
			ans = append(ans, v)
		}
	}
	return ans
}

func getcom(ans *[][]int, cand []int, tmp []int, k int) {
	if k == 0 {
		newtmp := make([]int, 0)
		newtmp = append(newtmp, tmp...)
		fmt.Println(newtmp)
		*ans = append(*ans, newtmp)
	}

	for i := 0; i < len(cand); i++ {
		tmp = append(tmp, cand[i])
		newcand := make([]int, 0)
		//newcand = append(newcand, cand[:i]...)
		newcand = append(newcand, cand[i+1:]...)
		getcom(ans, newcand, tmp, k-1)
		tmp = tmp[:len(tmp)-1]
	}
}

func combine(n int, k int) [][]int {
	ans := make([][]int, 0)
	cand := make([]int, n)
	for i := 0; i < n; i++ {
		cand[i] = i + 1
	}
	tmp := make([]int, 0)
	getcom(&ans, cand, tmp, k)
	return ans
}

func getsub(ans *[][]int, cand []int, tmp []int) {
	newtmp := make([]int, 0)
	newtmp = append(newtmp, tmp...)
	fmt.Println(newtmp)
	*ans = append(*ans, newtmp)

	for i := 0; i < len(cand); i++ {
		tmp = append(tmp, cand[i])
		newcand := make([]int, 0)
		//newcand = append(newcand, cand[:i]...)
		newcand = append(newcand, cand[i+1:]...)
		getsub(ans, newcand, tmp)
		tmp = tmp[:len(tmp)-1]
	}
}

func subsets(nums []int) [][]int {
	ans := make([][]int, 0)
	tmp := make([]int, 0)
	getsub(&ans, nums, tmp)
	return ans
}

func check_board(board [][]byte, i, j int, word string, haslen int) bool {
	if len(word) == haslen {
		return true
	}

	if i < 0 || j < 0 || i >= len(board) || j >= len(board[i]) {
		return false
	}
	if board[i][j] != word[haslen] {
		fmt.Println(board[i][j], word[0], board[i][j] != word[0])
		return false
	}
	ori := board[i][j]
	board[i][j] = ' '
	valid := check_board(board, i-1, j, word, haslen+1) || check_board(board, i, j-1, word, haslen+1) || check_board(board, i, j+1, word, haslen+1) || check_board(board, i+1, j, word, haslen+1)
	board[i][j] = ori
	return valid
}

func exist(board [][]byte, word string) bool {
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if check_board(board, i, j, word, 0) {
				return true
			}
		}
	}

	return false
}

func removeDuplicates(nums []int) int {
	i := 0
	for _, n := range nums {
		if i < 2 || n > nums[i-2] {
			nums[i] = n
			i++
		}
	}

	return i
}

func minWindow(s string, t string) string {
	m := make(map[byte]int)
	for i := 0; i < len(t); i++ {
		m[t[i]]++
	}
	count := len(t)
	begin, end := 0, 0
	d := math.MaxInt32
	head := 0
	for end < len(s) {
		if m[s[end]] > 0 {
			count--
		}
		m[s[end]]--
		end++
		for count == 0 {
			if end-begin < d {
				d = end - begin
				head = begin
			}
			if m[s[begin]] == 0 {
				count++
			}
			fmt.Println(m)
			m[s[begin]]++
			begin++
		}
	}
	if d == math.MaxInt32 {
		return ""
	} else {
		return s[head : d+head]
	}
}

/*func cal_cp(h []int,begin,end int)int{
    min := h[begin]
    for i:=begin+1; i <= end; i++{
        if h[i] < min{
            min = h[i]
        }
    }
    return min * (end-begin+1)
}



func largestRectangleArea(heights []int) int {
    l := len(heights)
    if l == 0{
        return 0
    }
    max := 0
    for i := 0 ; i < l ; i++{
        for j := i ; j < l ; j++{
            cp := cal_cp(heights,i,j)
            fmt.Println(i,j,cp)
            if cp > max{
                fmt.Println(i,j,cp)
                max = cp
            }
        }
    }
    return max
}*/

func largestRectangleArea(heights []int) int {
	l := len(heights)
	if l == 0 {
		return 0
	}
	ret := 0
	heights = append(heights, 0)
	index := MyS{}

	for i := 0; i < l+1; i++ {
		for !index.empty() && heights[index.back()] > heights[i] {
			h := heights[index.back()]
			index.pop()

			var sidx int
			if index.empty() {
				sidx = -1
			} else {
				sidx = index.back()
			}
			fmt.Println(h * (i - sidx - 1))
			if h*(i-sidx-1) > ret {
				ret = h * (i - sidx - 1)
			}
		}
		index.push(i)
		fmt.Println(index)
	}
	return ret
}

func isScramble(s1 string, s2 string) bool {
	fmt.Println(s1, s2)
	if s1 == s2 {
		return true
	}
	if len(s1) != len(s2) {
		return false
	}
	l := len(s1)
	arr := make([]int, 26)
	for i := 0; i < l; i++ {
		arr[s1[i]-'a']++
		arr[s2[i]-'a']--
	}
	for i := 0; i < 26; i++ {
		if arr[i] != 0 {
			return false
		}
	}
	for i := 1; i < l; i++ {
		if isScramble(s1[:i], s2[l-i:]) && isScramble(s1[i:], s2[:l-i]) {
			return true
		}
		if isScramble(s1[:i], s2[:i]) && isScramble(s1[i:], s2[i:]) {
			return true
		}
	}
	return false
}

func getsubdup(nums []int) [][]int {
	ans := make([][]int, 0)
	ans = append(ans, []int{})
	if len(nums) == 0 {
		return ans
	}
	for i := 0; i < len(nums); i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		tmpnum := make([]int, 0)
		//tmpnum = append(tmpnum,nums[:i]...)
		tmpnum = append(tmpnum, nums[i+1:]...)
		tmpans := getsubdup(tmpnum)
		for _, t := range tmpans {
			t = append(t, nums[i])
			ans = append(ans, t)
		}
	}

	return ans
}

func subsetsWithDup(nums []int) [][]int {
	sort(nums)
	return getsubdup(nums)
}

func numDecodings(s string) int {
	n := len(s)
	if n <= 0 {
		return 0
	}
	ans := make([]int, n+1)
	ans[0] = 1
	if s[0] == '0' {
		ans[1] = 0
	} else {
		ans[1] = 1
	}
	for i := 2; i <= n; i++ {
		first, _ := strconv.Atoi(s[i-1 : i])
		second, _ := strconv.Atoi(s[i-2 : i])
		if first >= 1 && first <= 9 {
			ans[i] += ans[i-1]
		}
		if second >= 10 && second <= 26 {
			ans[i] += ans[i-2]
		}
	}
	return ans[n]

}

func restoreIpAddresses(s string) []string {
	ans := make([]string, 0)
	l := len(s)
	//3 loops
	for i1 := 0; i1 < 4 && i1 < l-2; i1++ {
		for i2 := i1 + 1; i2 < i1+4 && i2 < l-1; i2++ {
			for i3 := i2 + 1; i3 < i2+4 && i3 < l; i3++ {
				s1, s2, s3, s4 := s[:i1], s[i1:i2], s[i2:i3], s[i3:]
				if validip(s1) && validip(s2) && validip(s3) && validip(s4) {
					ans = append(ans, s1+"."+s2+"."+s3+"."+s4)
				}
			}
		}
	}
	return ans
}

func validip(s string) bool {
	if len(s) == 0 || len(s) > 3 || (s[0] == '0' && len(s) > 1) {
		return false
	}
	num, _ := strconv.Atoi(s)
	if num > 255 {
		return false
	}
	return true
}

func isInterleave(s1 string, s2 string, s3 string) bool {
	if len(s3) == 0 {
		if len(s1) == 0 && len(s2) == 0 {
			return true
		} else {
			return false
		}
	}
	if len(s1) == 0 {
		if s2 == s3 {
			return true
		} else {
			return false
		}
	}
	if len(s2) == 0 {
		if s1 == s3 {
			return true
		} else {
			return false
		}
	}

	start1 := s1[0]
	start2 := s2[0]
	start3 := s3[0]

	return (start1 == start3 && isInterleave(s1[1:], s2, s3[1:])) || (start2 == start3 && isInterleave(s1, s2[1:], s3[1:]))

}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func ivb(root *TreeNode, min, max int64) bool {
	if root == nil {
		return true
	}
	if int64(root.Val) <= min || int64(root.Val) >= max {
		return false
	}
	return ivb(root.Left, min, int64(root.Val)) && ivb(root.Right, int64(root.Val), max)
}

func isValidBST(root *TreeNode) bool {
	return ivb(root, math.MinInt64, math.MaxInt64)
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
var first, second, pre *TreeNode

func recoverTree(root *TreeNode) {
	pre = &TreeNode{Val: math.MinInt32}
	traverse(root)
	tmp := first.Val
	first.Val = second.Val
	second.Val = tmp
}

func traverse(root *TreeNode) {
	if root == nil {
		return
	}
	traverse(root.Left)

	if first == nil && pre.Val >= root.Val {
		first = pre
	}
	if first != nil && pre.Val >= root.Val {
		second = root
	}
	pre = root
	traverse(root.Right)
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func lo(ans *[][]int, levelnode []*TreeNode) {
	if len(levelnode) == 0 {
		return
	}
	levelans := make([]int, 0)
	nextlevelnode := make([]*TreeNode, 0)
	for _, nd := range levelnode {
		levelans = append(levelans, nd.Val)
		if nd.Left != nil {
			nextlevelnode = append(nextlevelnode, nd.Left)
		}
		if nd.Right != nil {
			nextlevelnode = append(nextlevelnode, nd.Right)
		}
	}
	*ans = append(*ans, levelans)
	if len(nextlevelnode) != 0 {
		lo(ans, nextlevelnode)
	}
}
func levelOrder(root *TreeNode) [][]int {
	ans := make([][]int, 0)
	if root == nil {
		return ans

	}
	levelnode := []*TreeNode{root}
	lo(&ans, levelnode)
	return ans
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func zzlo(ans *[][]int, levelnode []*TreeNode, lr bool) {
	fmt.Println(lr)
	if len(levelnode) == 0 {
		return
	}
	levelans := make([]int, 0)
	nextlevelnode := make([]*TreeNode, 0)
	var idx int
	//if lr{
	//idx = 0
	//}else{
	idx = len(levelnode) - 1
	//}
	for idx >= 0 && idx < len(levelnode) {
		nd := levelnode[idx]
		levelans = append(levelans, nd.Val)
		if lr {
			if nd.Left != nil {
				nextlevelnode = append(nextlevelnode, nd.Left)
			}
			if nd.Right != nil {
				nextlevelnode = append(nextlevelnode, nd.Right)
			}
		} else {
			if nd.Right != nil {
				nextlevelnode = append(nextlevelnode, nd.Right)
			}

			if nd.Left != nil {
				nextlevelnode = append(nextlevelnode, nd.Left)
			}
		}
		//if lr{
		//idx ++
		//}else{
		idx--
		//}
	}
	*ans = append(*ans, levelans)
	if len(nextlevelnode) != 0 {
		zzlo(ans, nextlevelnode, !lr)
	}
}

func zigzagLevelOrder(root *TreeNode) [][]int {
	ans := make([][]int, 0)
	if root == nil {
		return ans

	}
	levelnode := []*TreeNode{root}
	zzlo(&ans, levelnode, true)
	return ans

}


/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
/*
func bthelper(prestart,instart,inend int,preorder,inorder []int)*TreeNode{
	if prestart > len(preorder)-1 || instart > inend{
		return nil
	}
	node := &TreeNode{Val:preorder[prestart]}
	var i int
	for i=instart; i<= inend; i++{
		if inorder[i] == preorder[prestart]{
			break
		}
	}
	node.Left = bthelper(prestart+1,instart,i-1,preorder,inorder)
	node.Right = bthelper(prestart+(i-instart+1),i+1,inend,preorder,inorder)

	return node
}
func buildTree(preorder []int, inorder []int) *TreeNode {
	return bthelper(0,0,len(inorder)-1,preorder,inorder)   
}*/

func bthelper(is,ie,ps,pe int,inorder,postorder []int)*TreeNode{
	if is > ie || ps > pe{
		return nil
	}
	node := &TreeNode{Val:postorder[pe]}
	var pos int
	for i:=is; i<= ie; i++{
		if inorder[i] == node.Val{
			pos = i
			break
		}
	}
	node.Left = bthelper(is,pos-1, ps, ps+(pos-is+1),inorder,postorder)
	node.Right = bthelper(pos+1,ie, pe-(ie-pos),pe-1,inorder,postorder)

	return node
}

func buildTree(inorder []int, postorder []int) *TreeNode {
	return bthelper(0,len(inorder)-1,0,len(postorder)-1,inorder,postorder)
}


func sltbhelper(head,tail *ListNode) *TreeNode{
	slow,fast := head,head
	if head == tail{
		return nil
	}
	for fast != tail && fast.Next != tail{
		fast = fast.Next.Next
		slow = slow.Next
	}
	root := &TreeNode{Val:slow.Val}
	root.Left = sltbhelper(head,slow)
	root.Right = sltbhelper(slow.Next,tail)
	return root
}

func sortedListToBST(head *ListNode) *TreeNode {
	if head == nil{
		return nil
	}else{
		return sltbhelper(head,nil)
	}
}

func findans(ans *[][]int,tmpans []int,root *TreeNode,sum int){
	if root == nil{
		return
	}
	tmpans = append(tmpans,root.Val)
	if root.Left == nil && root.Right == nil && sum - root.Val == 0{
		tmp := make([]int,0)
		tmp = append(tmp,tmpans...)
		*ans = append(*ans,tmp)
		return
	}
	findans(ans,tmpans,root.Left,sum-root.Val)
	findans(ans,tmpans,root.Right,sum-root.Val)
	tmpans = tmpans[:len(tmpans)-1]
}


func pathSum(root *TreeNode, sum int) [][]int {
	ans := make([][]int,0)
	tmpans := make([]int,0)
	findans(&ans,tmpans,root,sum)
	return ans
}


func flatten(root *TreeNode)  {
    if root == nil{
        return
    }
    flatten(root.Right)
    flatten(root.Left)
    root.Right = pre
    root.Left = nil
    pre = root
}

func numDistinct(s string, t string) int {
	sl := len(s)
	tl := len(t)
	mem := make([][]int,tl+1)
	for i:=0; i<=tl; i++{
		mem[i] = make([]int,sl+1)
	}
	for i:=0; i<=sl; i++{
		mem[0][i] = 1
	}
	for i:=0; i<tl; i++{
		for j:=0; j<sl; j++{
			if t[i] == s[j]{
				mem[i+1][j+1] = mem[i][j] + mem[i+1][j]
			}else{
				mem[i+1][j+1] = mem[i+1][j]
			}
		}
	}
	return mem[tl][sl]
}


func minimumTotal(triangle [][]int) int {
	height := len(triangle)
	minpath := triangle[height-1]
	for h := height-2; h>=0; h--{
		for i:=0; i<=h; i++{
			minpath[i] = min(minpath[i],minpath[i+1]) + triangle[h][i]
		}
	}
	return minpath[0]
}

var mv int

func maxPathSum(root *TreeNode) int {
	maxPathSumhelper(root)
	return mv
}
func maxPathSumhelper(root *TreeNode) int {
	if root == nil{
		return 0
	}
	left := max(0,maxPathSumhelper(root.Left))
	right := max(0,maxPathSumhelper(root.Right))
	mv = max(mv,left+right+root.Val)
	return max(left,right) + root.Val
}

var min_len int


func ladderLength(beginWord string, endWord string, wordList []string) int {
	min_len = 0
	llhelper(endWord,[]string{beginWord},wordList,1)
	return min_len
}

func isOneDiff(s1,s2 string)bool{
	diff := 0
	for i:=0; i < len(s1); i++{
		if s1[i] != s2[i]{
			diff ++
		}
	}
	return diff == 1
}

func llhelper(end string,words []string,wordList []string,level int){
	fmt.Println("level",words)
	//fmt.Println("words",wordList)
	newwords := make([]string,0)
	if len(words) == 0 || len(wordList) == 0{
		return
	}
	for _,begin := range words{
		for i:=0; i < len(wordList);{
			if isOneDiff(begin,wordList[i]){

				if end == wordList[i]{
					min_len = level+1
					return
				}
				newwords = append(newwords,wordList[i])
				wordList = append(wordList[:i],wordList[i+1:]...)
			}else{
				i++
			}
		}
	}
	llhelper(end,newwords,wordList,level+1)
}

func present(ws []string,w string)bool{
	for _,v := range ws{
		if v == w{
			return true
		}
	}
	return false
}

func flhelper(end string,words [][]string,wordList []string,ans *[][]string,visited *[]bool){
        fmt.Println("level",len(words))
        fmt.Println("word",len(wordList))
        //fmt.Println("level",words)
        //fmt.Println("word",wordList)
        newwords := make([][]string,0)
        if len(words) == 0 || len(wordList) == 0{
                return
        }
        find := false
	//visited := make([]string,0)
	thisvisit := make([]int,0)
	calued := make(map[string][]int)
        for _,beginlist := range words{
                begin := beginlist[len(beginlist)-1]
		_,ispresent := calued[begin]
		if !ispresent{
			thislist := make([]int,0)
                	for i:=0; i < len(wordList);{
                	        if !present(beginlist,wordList[i]) && isOneDiff(begin,wordList[i]){
                	                newpath := make([]string,0)
                	                newpath = append(newpath,beginlist...)
                	                newpath = append(newpath,wordList[i])
					thisvisit = append(thisvisit,i)
					thislist = append(thislist,i)
                	                if end == wordList[i]{
                	                        find = true
                	                        *ans = append(*ans,newpath)
                	                        i++
                	                        continue
                	                }
					if !((*visited)[i]){
                	                	newwords = append(newwords,newpath)
					}
                	                //wordList = append(wordList[:i],wordList[i+1:]...)
                	        }
                	        i++

                	}
			calued[begin] = thislist
		}else{
			for _,i := range calued[begin]{
			        if !present(beginlist,wordList[i]){
                	                newpath := make([]string,0)
                	                newpath = append(newpath,beginlist...)
                	                newpath = append(newpath,wordList[i])
					thisvisit = append(thisvisit,i)
                	                if end == wordList[i]{
                	                        find = true
                	                        *ans = append(*ans,newpath)
                	                        continue
                	                }
					if !((*visited)[i]){
                	                	newwords = append(newwords,newpath)
					}
                	                //wordList = append(wordList[:i],wordList[i+1:]...)
                	        }

			}
		
		}
        }
	for _,vi := range thisvisit{
		(*visited)[vi] = true
	}

        if find{
                return
        }
        flhelper(end,newwords,wordList,ans,visited)
}



func findLadders(beginWord string, endWord string, wordList []string) [][]string {
	ans := make([][]string,0)
	start := make([][]string,1)
	start[0] = []string{beginWord}
	visited  := make([]bool,len(wordList))
	flhelper(endWord,start,wordList,&ans,&visited)
	return ans
}


func main() {
	//fmt.Println(findLadders("hit","cog",[]string{"hot","dot","dog","lot","log","cog"}))
	//fmt.Println(findLadders("red","tax",[]string{"ted","tex","red","tax","tad","den","rex","pee"}))
	fmt.Println(findLadders("nanny","aloud",[]string{"ricky","grind","cubic","panic","lover","farce","gofer","sales","flint","omens","lipid","briny","cloth","anted","slime","oaten","harsh","touts","stoop","cabal","lazed","elton","skunk","nicer","pesky","kusch","bused","kinda","tunis","enjoy","aches","prowl","babar","rooms","burst","slush","pines","urine","pinky","bayed","mania","light","flare","wares","women","verne","moron","shine","bluer","zeros","bleak","brief","tamra","vasts","jamie","lairs","penal","worst","yowls","pills","taros","addle","alyce","creep","saber","floyd","cures","soggy","vexed","vilma","cabby","verde","euler","cling","wanna","jenny","donor","stole","sakha","blake","sanes","riffs","forge","horus","sered","piked","prosy","wases","glove","onset","spake","benin","talks","sites","biers","wendy","dante","allan","haven","nears","shaka","sloth","perky","spear","spend","clint","dears","sadly","units","vista","hinds","marat","natal","least","bough","pales","boole","ditch","greys","slunk","bitch","belts","sense","skits","monty","yawns","music","hails","alien","gibes","lille","spacy","argot","wasps","drubs","poops","bella","clone","beast","emend","iring","start","darla","bells","cults","dhaka","sniff","seers","bantu","pages","fever","tacky","hoses","strop","climb","pairs","later","grant","raven","stael","drips","lucid","awing","dines","balms","della","galen","toned","snips","shady","chili","fears","nurse","joint","plump","micky","lions","jamal","queer","ruins","frats","spoof","semen","pulps","oldie","coors","rhone","papal","seals","spans","scaly","sieve","klaus","drums","tided","needs","rider","lures","treks","hares","liner","hokey","boots","primp","laval","limes","putts","fonda","damon","pikes","hobbs","specs","greet","ketch","braid","purer","tsars","berne","tarts","clean","grate","trips","chefs","timex","vicky","pares","price","every","beret","vices","jodie","fanny","mails","built","bossy","farms","pubic","gongs","magma","quads","shell","jocks","woods","waded","parka","jells","worse","diner","risks","bliss","bryan","terse","crier","incur","murky","gamed","edges","keens","bread","raced","vetch","glint","zions","porno","sizes","mends","ached","allie","bands","plank","forth","fuels","rhyme","wimpy","peels","foggy","wings","frill","edgar","slave","lotus","point","hints","germs","clung","limed","loafs","realm","myron","loopy","plush","volts","bimbo","smash","windy","sours","choke","karin","boast","whirr","tiber","dimes","basel","cutes","pinto","troll","thumb","decor","craft","tared","split","josue","tramp","screw","label","lenny","apses","slept","sikhs","child","bouts","cites","swipe","lurks","seeds","fists","hoard","steed","reams","spoil","diego","peale","bevel","flags","mazes","quart","snipe","latch","lards","acted","falls","busby","holed","mummy","wrong","wipes","carlo","leers","wails","night","pasty","eater","flunk","vedas","curse","tyros","mirth","jacky","butte","wired","fixes","tares","vague","roved","stove","swoon","scour","coked","marge","cants","comic","corns","zilch","typos","lives","truer","comma","gaily","teals","witty","hyper","croat","sways","tills","hones","dowel","llano","clefs","fores","cinch","brock","vichy","bleed","nuder","hoyle","slams","macro","arabs","tauts","eager","croak","scoop","crime","lurch","weals","fates","clipt","teens","bulls","domed","ghana","culls","frame","hanky","jared","swain","truss","drank","lobby","lumps","pansy","whews","saris","trite","weeps","dozes","jeans","flood","chimu","foxes","gelds","sects","scoff","poses","mares","famed","peers","hells","laked","zests","wring","steal","snoot","yodel","scamp","ellis","bandy","marry","jives","vises","blurb","relay","patch","haley","cubit","heine","place","touch","grain","gerry","badly","hooke","fuchs","savor","apron","judge","loren","britt","smith","tammy","altar","duels","huber","baton","dived","apace","sedan","basts","clark","mired","perch","hulks","jolly","welts","quack","spore","alums","shave","singe","lanny","dread","profs","skeet","flout","darin","newed","steer","taine","salvo","mites","rules","crash","thorn","olive","saves","yawed","pique","salon","ovens","dusty","janie","elise","carve","winds","abash","cheep","strap","fared","discs","poxed","hoots","catch","combo","maize","repay","mario","snuff","delve","cored","bards","sudan","shuns","yukon","jowls","wayne","torus","gales","creek","prove","needy","wisps","terri","ranks","books","dicky","tapes","aping","padre","roads","nines","seats","flats","rains","moira","basic","loves","pulls","tough","gills","codes","chest","teeny","jolts","woody","flame","asked","dulls","hotly","glare","mucky","spite","flake","vines","lindy","butts","froth","beeps","sills","bunny","flied","shaun","mawed","velds","voled","doily","patel","snake","thigh","adler","calks","desks","janus","spunk","baled","match","strip","hosed","nippy","wrest","whams","calfs","sleet","wives","boars","chain","table","duked","riped","edens","galas","huffs","biddy","claps","aleut","yucks","bangs","quids","glenn","evert","drunk","lusts","senna","slate","manet","roted","sleep","loxes","fluky","fence","clamp","doted","broad","sager","spark","belch","mandy","deana","beyer","hoist","leafy","levee","libel","tonic","aloes","steam","skews","tides","stall","rifts","saxon","mavis","asama","might","dotes","tangs","wroth","kited","salad","liens","clink","glows","balky","taffy","sided","sworn","oasis","tenth","blurt","tower","often","walsh","sonny","andes","slump","scans","boded","chive","finer","ponce","prune","sloes","dined","chums","dingo","harte","ahead","event","freer","heart","fetch","sated","soapy","skins","royal","cuter","loire","minot","aisle","horny","slued","panel","eight","snoop","pries","clive","pored","wrist","piped","daren","cells","parks","slugs","cubed","highs","booze","weary","stain","hoped","finny","weeds","fetid","racer","tasks","right","saint","shahs","basis","refer","chart","seize","lulls","slant","belay","clots","jinny","tours","modes","gloat","dunks","flute","conch","marts","aglow","gayer","lazes","dicks","chime","bears","sharp","hatch","forms","terry","gouda","thins","janet","tonya","axons","sewed","danny","rowdy","dolts","hurry","opine","fifty","noisy","spiky","humid","verna","poles","jayne","pecos","hooky","haney","shams","snots","sally","ruder","tempe","plunk","shaft","scows","essie","dated","fleet","spate","bunin","hikes","sodas","filly","thyme","fiefs","perks","chary","kiths","lidia","lefty","wolff","withe","three","crawl","wotan","brown","japed","tolls","taken","threw","crave","clash","layer","tends","notes","fudge","musky","bawdy","aline","matts","shirr","balks","stash","wicks","crepe","foods","fares","rotes","party","petty","press","dolly","mangy","leeks","silly","leant","nooks","chapt","loose","caged","wages","grist","alert","sheri","moody","tamps","hefts","souls","rubes","rolex","skulk","veeps","nonce","state","level","whirl","bight","grits","reset","faked","spiny","mixes","hunks","major","missy","arius","damns","fitly","caped","mucus","trace","surat","lloyd","furry","colin","texts","livia","reply","twill","ships","peons","shear","norms","jumbo","bring","masks","zippy","brine","dorks","roded","sinks","river","wolfs","strew","myths","pulpy","prank","veins","flues","minus","phone","banns","spell","burro","brags","boyle","lambs","sides","knees","clews","aired","skirt","heavy","dimer","bombs","scums","hayes","chaps","snugs","dusky","loxed","ellen","while","swank","track","minim","wiled","hazed","roofs","cantu","sorry","roach","loser","brass","stint","jerks","dirks","emory","campy","poise","sexed","gamer","catty","comte","bilbo","fasts","ledge","drier","idles","doors","waged","rizal","pured","weirs","crisp","tasty","sored","palmy","parts","ethel","unify","crows","crest","udder","delis","punks","dowse","totes","emile","coded","shops","poppa","pours","gushy","tiffs","shads","birds","coils","areas","boons","hulls","alter","lobes","pleat","depth","fires","pones","serra","sweat","kline","malay","ruled","calve","tired","drabs","tubed","wryer","slung","union","sonya","aided","hewed","dicey","grids","nixed","whits","mills","buffs","yucky","drops","ready","yuppy","tweet","napes","cadre","teach","rasps","dowdy","hoary","canto","posed","dumbo","kooks","reese","snaky","binge","byron","phony","safer","friar","novel","scale","huron","adorn","carla","fauna","myers","hobby","purse","flesh","smock","along","boils","pails","times","panza","lodge","clubs","colby","great","thing","peaks","diana","vance","whets","bergs","sling","spade","soaks","beach","traps","aspen","romps","boxed","fakir","weave","nerds","swazi","dotty","curls","diver","jonas","waite","verbs","yeast","lapel","barth","soars","hooks","taxed","slews","gouge","slags","chang","chafe","saved","josie","syncs","fonds","anion","actor","seems","pyrex","isiah","glued","groin","goren","waxes","tonia","whine","scads","knelt","teaks","satan","tromp","spats","merry","wordy","stake","gland","canal","donna","lends","filed","sacks","shied","moors","paths","older","pooch","balsa","riced","facet","decaf","attic","elder","akron","chomp","chump","picky","money","sheer","bolls","crabs","dorms","water","veers","tease","dummy","dumbs","lethe","halls","rifer","demon","fucks","whips","plops","fuses","focal","taces","snout","edict","flush","burps","dawes","lorry","spews","sprat","click","deann","sited","aunts","quips","godly","pupil","nanny","funks","shoon","aimed","stacy","helms","mints","banks","pinch","local","twine","pacts","deers","halos","slink","preys","potty","ruffs","pusan","suits","finks","slash","prods","dense","edsel","heeds","palls","slats","snits","mower","rares","ailed","rouge","ellie","gated","lyons","duded","links","oaths","letha","kicks","firms","gravy","month","kongo","mused","ducal","toted","vocal","disks","spied","studs","macao","erick","coupe","starr","reaps","decoy","rayon","nicks","breed","cosby","haunt","typed","plain","trays","muled","saith","drano","cower","snows","buses","jewry","argus","doers","flays","swish","resin","boobs","sicks","spies","bails","wowed","mabel","check","vapid","bacon","wilda","ollie","loony","irked","fraud","doles","facts","lists","gazed","furls","sunks","stows","wilde","brick","bowed","guise","suing","gates","niter","heros","hyped","clomp","never","lolls","rangy","paddy","chant","casts","terns","tunas","poker","scary","maims","saran","devon","tripe","lingo","paler","coped","bride","voted","dodge","gross","curds","sames","those","tithe","steep","flaks","close","swops","stare","notch","prays","roles","crush","feuds","nudge","baned","brake","plans","weepy","dazed","jenna","weiss","tomes","stews","whist","gibed","death","clank","cover","peeks","quick","abler","daddy","calls","scald","lilia","flask","cheer","grabs","megan","canes","jules","blots","mossy","begun","freak","caved","hello","hades","theed","wards","darcy","malta","peter","whorl","break","downs","odder","hoofs","kiddo","macho","fords","liked","flees","swing","elect","hoods","pluck","brook","astir","bland","sward","modal","flown","ahmad","waled","craps","cools","roods","hided","plath","kings","grips","gives","gnats","tabby","gauls","think","bully","fogey","sawed","lints","pushy","banes","drake","trail","moral","daley","balds","chugs","geeky","darts","soddy","haves","opens","rends","buggy","moles","freud","gored","shock","angus","puree","raves","johns","armed","packs","minis","reich","slots","totem","clown","popes","brute","hedge","latin","stoke","blend","pease","rubik","greer","hindi","betsy","flows","funky","kelli","humps","chewy","welds","scowl","yells","cough","sasha","sheaf","jokes","coast","words","irate","hales","camry","spits","burma","rhine","bends","spill","stubs","power","voles","learn","knoll","style","twila","drove","dacca","sheen","papas","shale","jones","duped","tunny","mouse","floss","corks","skims","swaps","inned","boxer","synch","skies","strep","bucks","belau","lower","flaky","quill","aural","rufus","floes","pokes","sends","sates","dally","boyer","hurts","foyer","gowns","torch","luria","fangs","moats","heinz","bolts","filet","firth","begot","argue","youth","chimp","frogs","kraft","smite","loges","loons","spine","domes","pokey","timur","noddy","doggy","wades","lanes","hence","louts","turks","lurid","goths","moist","bated","giles","stood","winos","shins","potts","brant","vised","alice","rosie","dents","babes","softy","decay","meats","tanya","rusks","pasts","karat","nuked","gorge","kinks","skull","noyce","aimee","watch","cleat","stuck","china","testy","doses","safes","stage","bayes","twins","limps","denis","chars","flaps","paces","abase","grays","deans","maria","asset","smuts","serbs","whigs","vases","robyn","girls","pents","alike","nodal","molly","swigs","swill","slums","rajah","bleep","beget","thanh","finns","clock","wafts","wafer","spicy","sorer","reach","beats","baker","crown","drugs","daisy","mocks","scots","fests","newer","agate","drift","marta","chino","flirt","homed","bribe","scram","bulks","servo","vesta","divas","preps","naval","tally","shove","ragas","blown","droll","tryst","lucky","leech","lines","sires","pyxed","taper","trump","payee","midge","paris","bored","loads","shuts","lived","swath","snare","boned","scars","aeons","grime","writs","paige","rungs","blent","signs","davis","dials","daubs","rainy","fawns","wrier","golds","wrath","ducks","allow","hosea","spike","meals","haber","muses","timed","broom","burks","louis","gangs","pools","vales","altai","elope","plied","slain","chasm","entry","slide","bawls","title","sings","grief","viola","doyle","peach","davit","bench","devil","latex","miles","pasha","tokes","coves","wheel","tried","verdi","wanda","sivan","prior","fryer","plots","kicky","porch","shill","coats","borne","brink","pawed","erwin","tense","stirs","wends","waxen","carts","smear","rival","scare","phase","bragg","crane","hocks","conan","bests","dares","molls","roots","dunes","slips","waked","fours","bolds","slosh","yemen","poole","solid","ports","fades","legal","cedes","green","curie","seedy","riper","poled","glade","hosts","tools","razes","tarry","muddy","shims","sword","thine","lasts","bloat","soled","tardy","foots","skiff","volta","murks","croci","gooks","gamey","pyxes","poems","kayla","larva","slaps","abuse","pings","plows","geese","minks","derby","super","inked","manic","leaks","flops","lajos","fuzes","swabs","twigs","gummy","pyres","shrew","islet","doled","wooly","lefts","hunts","toast","faith","macaw","sonia","leafs","colas","conks","altos","wiped","scene","boors","patsy","meany","chung","wakes","clear","ropes","tahoe","zones","crate","tombs","nouns","garth","puked","chats","hanks","baked","binds","fully","soaps","newel","yarns","puers","carps","spelt","lully","towed","scabs","prime","blest","patty","silky","abner","temps","lakes","tests","alias","mines","chips","funds","caret","splat","perry","turds","junks","cramp","saned","peary","snarl","fired","stung","nancy","bulge","styli","seams","hived","feast","triad","jaded","elvin","canny","birth","routs","rimed","pusey","laces","taste","basie","malls","shout","prier","prone","finis","claus","loops","heron","frump","spare","menus","ariel","crams","bloom","foxed","moons","mince","mixed","piers","deres","tempt","dryer","atone","heats","dario","hawed","swims","sheet","tasha","dings","clare","aging","daffy","wried","foals","lunar","havel","irony","ronny","naves","selma","gurus","crust","percy","murat","mauro","cowed","clang","biker","harms","barry","thump","crude","ulnae","thong","pager","oases","mered","locke","merle","soave","petal","poser","store","winch","wedge","inlet","nerdy","utter","filth","spray","drape","pukes","ewers","kinds","dates","meier","tammi","spoor","curly","chill","loped","gooey","boles","genet","boost","beets","heath","feeds","growl","livid","midst","rinds","fresh","waxed","yearn","keeps","rimes","naked","flick","plies","deeps","dirty","hefty","messy","hairy","walks","leper","sykes","nerve","rover","jived","brisk","lenin","viper","chuck","sinus","luger","ricks","hying","rusty","kathy","herds","wider","getty","roman","sandy","pends","fezes","trios","bites","pants","bless","diced","earth","shack","hinge","melds","jonah","chose","liver","salts","ratty","ashed","wacky","yokes","wanly","bruce","vowel","black","grail","lungs","arise","gluts","gluey","navel","coyer","ramps","miter","aldan","booth","musty","rills","darns","tined","straw","kerri","hared","lucks","metes","penny","radon","palms","deeds","earls","shard","pried","tampa","blank","gybes","vicki","drool","groom","curer","cubes","riggs","lanky","tuber","caves","acing","golly","hodge","beard","ginny","jibed","fumes","astor","quito","cargo","randi","gawky","zings","blind","dhoti","sneak","fatah","fixer","lapps","cline","grimm","fakes","maine","erika","dealt","mitch","olden","joist","gents","likes","shelf","silts","goats","leads","marin","spire","louie","evans","amuse","belly","nails","snead","model","whats","shari","quote","tacks","nutty","lames","caste","hexes","cooks","miner","shawn","anise","drama","trike","prate","ayers","loans","botch","vests","cilia","ridge","thugs","outed","jails","moped","plead","tunes","nosed","wills","lager","lacks","cried","wince","berle","flaws","boise","tibet","bided","shred","cocky","brice","delta","congo","holly","hicks","wraps","cocks","aisha","heard","cured","sades","horsy","umped","trice","dorky","curve","ferry","haler","ninth","pasta","jason","honer","kevin","males","fowls","awake","pores","meter","skate","drink","pussy","soups","bases","noyes","torts","bogus","still","soupy","dance","worry","eldon","stern","menes","dolls","dumpy","gaunt","grove","coops","mules","berry","sower","roams","brawl","greed","stags","blurs","swift","treed","taney","shame","easel","moves","leger","ville","order","spock","nifty","brian","elias","idler","serve","ashen","bizet","gilts","spook","eaten","pumas","cotes","broke","toxin","groan","laths","joins","spots","hated","tokay","elite","rawer","fiats","cards","sassy","milks","roost","glean","lutes","chins","drown","marks","pined","grace","fifth","lodes","rusts","terms","maxes","savvy","choir","savoy","spoon","halve","chord","hulas","sarah","celia","deems","ninny","wines","boggy","birch","raved","wales","beams","vibes","riots","warty","nigel","askew","faxes","sedge","sheol","pucks","cynic","relax","boers","whims","bents","candy","luann","slogs","bonny","barns","iambs","fused","duffy","guilt","bruin","pawls","penis","poppy","owing","tribe","tuner","moray","timid","ceded","geeks","kites","curio","puffy","perot","caddy","peeve","cause","dills","gavel","manse","joker","lynch","crank","golda","waits","wises","hasty","paves","grown","reedy","crypt","tonne","jerky","axing","swept","posse","rings","staff","tansy","pared","glaze","grebe","gonna","shark","jumps","vials","unset","hires","tying","lured","motes","linen","locks","mamas","nasty","mamie","clout","nader","velma","abate","tight","dales","serer","rives","bales","loamy","warps","plato","hooch","togae","damps","ofter","plumb","fifes","filmy","wiper","chess","lousy","sails","brahe","ounce","flits","hindu","manly","beaux","mimed","liken","forts","jambs","peeps","lelia","brews","handy","lusty","brads","marne","pesos","earle","arson","scout","showy","chile","sumps","hiked","crook","herbs","silks","alamo","mores","dunce","blaze","stank","haste","howls","trots","creon","lisle","pause","hates","mulch","mined","moder","devin","types","cindy","beech","tuned","mowed","pitts","chaos","colds","bidet","tines","sighs","slimy","brain","belle","leery","morse","ruben","prows","frown","disco","regal","oaken","sheds","hives","corny","baser","fated","throe","revel","bores","waved","shits","elvia","ferns","maids","color","coifs","cohan","draft","hmong","alton","stine","cluck","nodes","emily","brave","blair","blued","dress","bunts","holst","clogs","rally","knack","demos","brady","blues","flash","goofy","blocs","diane","colic","smile","yules","foamy","splay","bilge","faker","foils","condo","knell","crack","gallo","purls","auras","cakes","doves","joust","aides","lades","muggy","tanks","middy","tarps","slack","capet","frays","donny","venal","yeats","misty","denim","glass","nudes","seeps","gibbs","blows","bobbi","shane","yards","pimps","clued","quiet","witch","boxes","prawn","kerry","torah","kinko","dingy","emote","honor","jelly","grins","trope","vined","bagel","arden","rapid","paged","loved","agape","mural","budge","ticks","suers","wendi","slice","salve","robin","bleat","batik","myles","teddy","flatt","puppy","gelid","largo","attar","polls","glide","serum","fundy","sucks","shalt","sewer","wreak","dames","fonts","toxic","hines","wormy","grass","louse","bowls","crass","benny","moire","margo","golfs","smart","roxie","wight","reign","dairy","clops","paled","oddly","sappy","flair","shown","bulgy","benet","larch","curry","gulfs","fends","lunch","dukes","doris","spoke","coins","manna","conga","jinns","eases","dunno","tisha","swore","rhino","calms","irvin","clans","gully","liege","mains","besot","serge","being","welch","wombs","draco","lynda","forty","mumps","bloch","ogden","knits","fussy","alder","danes","loyal","valet","wooer","quire","liefs","shana","toyed","forks","gages","slims","cloys","yates","rails","sheep","nacho","divan","honks","stone","snack","added","basal","hasps","focus","alone","laxes","arose","lamed","wrapt","frail","clams","plait","hover","tacos","mooch","fault","teeth","marva","mucks","tread","waves","purim","boron","horde","smack","bongo","monte","swirl","deals","mikes","scold","muter","sties","lawns","fluke","jilts","meuse","fives","sulky","molds","snore","timmy","ditty","gasps","kills","carey","jawed","byers","tommy","homer","hexed","dumas","given","mewls","smelt","weird","speck","merck","keats","draws","trent","agave","wells","chews","blabs","roves","grieg","evens","alive","mulls","cared","garbo","fined","happy","trued","rodes","thurs","cadet","alvin","busch","moths","guild","staci","lever","widen","props","hussy","lamer","riley","bauer","chirp","rants","poxes","shyer","pelts","funny","slits","tinge","ramos","shift","caper","credo","renal","veils","covey","elmer","mated","tykes","wooed","briar","gears","foley","shoes","decry","hypes","dells","wilds","runts","wilts","white","easts","comer","sammy","lochs","favor","lance","dawns","bushy","muted","elsie","creel","pocks","tenet","cagey","rides","socks","ogled","soils","sofas","janna","exile","barks","frank","takes","zooms","hakes","sagan","scull","heaps","augur","pouch","blare","bulbs","wryly","homey","tubas","limbo","hardy","hoagy","minds","bared","gabby","bilks","float","limns","clasp","laura","range","brush","tummy","kilts","cooed","worms","leary","feats","robes","suite","veals","bosch","moans","dozen","rarer","slyer","cabin","craze","sweet","talon","treat","yanks","react","creed","eliza","sluts","cruet","hafts","noise","seder","flies","weeks","venus","backs","eider","uriel","vouch","robed","hacks","perth","shiny","stilt","torte","throb","merer","twits","reeds","shawl","clara","slurs","mixer","newts","fried","woolf","swoop","kaaba","oozed","mayer","caned","laius","lunge","chits","kenny","lifts","mafia","sowed","piled","stein","whack","colts","warms","cleft","girds","seeks","poets","angel","trade","parsi","tiers","rojas","vexes","bryce","moots","grunt","drain","lumpy","stabs","poohs","leapt","polly","cuffs","giddy","towns","dacha","quoth","provo","dilly","carly","mewed","tzars","crock","toked","speak","mayas","pssts","ocher","motel","vogue","camps","tharp","taunt","drone","taint","badge","scott","scats","bakes","antes","gruel","snort","capes","plate","folly","adobe","yours","papaw","hench","moods","clunk","chevy","tomas","narcs","vonda","wiles","prigs","chock","laser","viced","stiff","rouse","helps","knead","gazer","blade","tumid","avail","anger","egged","guide","goads","rabin","toddy","gulps","flank","brats","pedal","junky","marco","tinny","tires","flier","satin","darth","paley","gumbo","rared","muffs","rower","prude","frees","quays","homes","munch","beefs","leash","aston","colon","finch","bogey","leaps","tempo","posts","lined","gapes","locus","maori","nixes","liven","songs","opted","babel","wader","barer","farts","lisps","koran","lathe","trill","smirk","mamma","viler","scurf","ravel","brigs","cooky","sachs","fulls","goals","turfs","norse","hauls","cores","fairy","pluto","kneed","cheek","pangs","risen","czars","milne","cribs","genes","wefts","vents","sages","seres","owens","wiley","flume","haded","auger","tatty","onion","cater","wolfe","magic","bodes","gulls","gazes","dandy","snags","rowed","quell","spurn","shore","veldt","turns","slavs","coach","stalk","snuck","piles","orate","joyed","daily","crone","wager","solos","earns","stark","lauds","kasey","villa","gnaws","scent","wears","fains","laced","tamer","pipes","plant","lorie","rivet","tamed","cozen","theme","lifer","sunny","shags","flack","gassy","eased","jeeps","shire","fargo","timer","brash","behan","basin","volga","krone","swiss","docks","booed","ebert","gusty","delay","oared","grady","buick","curbs","crete","lucas","strum","besom","gorse","troth","donne","chink","faced","ahmed","texas","longs","aloud","bethe","cacao","hilda","eagle","karyn","harks","adder","verse","drays","cello","taped","snide","taxis","kinky","penes","wicca","sonja","aways","dyers","bolas","elfin","slope","lamps","hutch","lobed","baaed","masts","ashes","ionic","joyce","payed","brays","malts","dregs","leaky","runny","fecal","woven","hurls","jorge","henna","dolby","booty","brett","dykes","rural","fight","feels","flogs","brunt","preen","elvis","dopey","gripe","garry","gamma","fling","space","mange","storm","arron","hairs","rogue","repel","elgar","ruddy","cross","medan","loses","howdy","foams","piker","halts","jewel","avery","stool","cruel","cases","ruses","cathy","harem","flour","meted","faces","hobos","charm","jamar","cameo","crape","hooey","reefs","denny","mitts","sores","smoky","nopes","sooty","twirl","toads","vader","julep","licks","arias","wrote","north","bunks","heady","batch","snaps","claws","fouls","faded","beans","wimps","idled","pulse","goons","noose","vowed","ronda","rajas","roast","allah","punic","slows","hours","metal","slier","meaty","hanna","curvy","mussy","truth","troys","block","reels","print","miffs","busts","bytes","cream","otter","grads","siren","kilos","dross","batty","debts","sully","bares","baggy","hippy","berth","gorky","argon","wacko","harry","smoke","fails","perms","score","steps","unity","couch","kelly","rumps","fines","mouth","broth","knows","becky","quits","lauri","trust","grows","logos","apter","burrs","zincs","buyer","bayer","moose","overt","croon","ousts","lands","lithe","poach","jamel","waive","wiser","surly","works","paine","medal","glads","gybed","paint","lorre","meant","smugs","bryon","jinni","sever","viols","flubs","melts","heads","peals","aiken","named","teary","yalta","styes","heist","bongs","slops","pouts","grape","belie","cloak","rocks","scone","lydia","goofs","rents","drive","crony","orlon","narks","plays","blips","pence","march","alger","baste","acorn","billy","croce","boone","aaron","slobs","idyls","irwin","elves","stoat","doing","globe","verve","icons","trial","olsen","pecks","there","blame","tilde","milky","sells","tangy","wrack","fills","lofty","truce","quark","delia","stowe","marty","overs","putty","coral","swine","stats","swags","weans","spout","bulky","farsi","brest","gleam","beaks","coons","hater","peony","huffy","exert","clips","riven","payer","doped","salas","meyer","dryad","thuds","tilts","quilt","jetty","brood","gulch","corps","tunic","hubby","slang","wreck","purrs","punch","drags","chide","sulks","tints","huger","roped","dopes","booby","rosin","outer","gusto","tents","elude","brows","lease","ceres","laxer","worth","necks","races","corey","trait","stuns","soles","teems","scrip","privy","sight","minor","alisa","stray","spank","cress","nukes","rises","gusts","aurae","karma","icing","prose","biked","grand","grasp","skein","shaky","clump","rummy","stock","twain","zoned","offed","ghats","mover","randy","vault","craws","thees","salem","downy","sangs","chore","cited","grave","spinx","erica","raspy","dying","skips","clerk","paste","moved","rooks","intel","moses","avers","staid","yawls","blast","lyres","monks","gaits","floor","saner","waver","assam","infer","wands","bunch","dryly","weedy","honey","baths","leach","shorn","shows","dream","value","dooms","spiro","raped","shook","stead","moran","ditto","loots","tapir","looms","clove","stops","pinks","soppy","ripen","wench","shone","bauds","doric","leans","nadia","cries","camus","boozy","maris","fools","morns","bides","greek","gauss","roget","lamar","hazes","beefy","dupes","refed","felts","larry","guile","ables","wants","warns","toils","bathe","edger","paced","rinks","shoos","erich","whore","tiger","jumpy","lamas","stack","among","punts","scalp","alloy","solon","quite","comas","whole","parse","tries","reeve","tiled","deena","roomy","rodin","aster","twice","musts","globs","parch","drawn","filch","bonds","tells","droop","janis","holds","scant","lopes","based","keven","whiny","aspic","gains","franz","jerri","steel","rowel","vends","yelps","begin","logic","tress","sunni","going","barge","blood","burns","basks","waifs","bones","skill","hewer","burly","clime","eking","withs","capek","berta","cheap","films","scoot","tweed","sizer","wheat","acton","flung","ponds","tracy","fiver","berra","roger","mutes","burke","miked","valve","whisk","runes","parry","toots","japes","roars","rough","irons","romeo","cages","reeks","cigar","saiph","dully","hangs","chops","rolls","prick","acuff","spent","sulla","train","swell","frets","names","anita","crazy","sixth","blunt","fewer","large","brand","slick","spitz","rears","ogres","toffy","yolks","flock","gnawn","eries","blink","skier","feted","tones","snail","ether","barbs","noses","hears","upset","awash","cloud","trunk","degas","dungs","rated","shall","yeahs","coven","sands","susan","fable","gunny","began","serfs","balls","dinky","madge","prong","spilt","lilly","brawn","comet","spins","raids","dries","sorts","makes","mason","mayra","royce","stout","mealy","pagan","nasal","folds","libby","coups","photo","mosey","amens","speed","lords","board","fetal","lagos","scope","raked","bonus","mutts","willy","sport","bingo","thant","araby","bette","rebel","gases","small","humus","grosz","beset","slays","steve","scrap","blahs","south","pride","heels","tubes","beady","lacey","genus","mauls","vying","spice","sexes","ester","drams","today","comae","under","jests","direr","yoked","tempi","early","boats","jesus","warts","guppy","gilda","quota","token","edwin","ringo","gaped","lemon","hurst","manor","arrow","mists","prize","silas","blobs","diets","ervin","stony","buddy","bates","rabid","ducat","ewing","jaunt","beads","doyen","blush","thoth","tiles","piper","short","peron","alley","decks","shunt","whirs","cushy","roils","betty","plugs","woken","jibes","foray","merak","ruing","becks","whale","shoot","dwelt","spawn","fairs","dozed","celts","blond","tikes","sabin","feint","vamps","cokes","willa","slues","bills","force","curst","yokel","surer","miler","fices","arced","douse","hilly","lucio","tongs","togas","minty","sagas","pates","welsh","bruno","decal","elate","linux","gyros","pryor","mousy","pains","shake","spica","pupal","probe","mount","shirk","purus","kilns","rests","graze","hague","spuds","sweep","momma","burch","maces","samar","brace","riser","booms","build","camel","flyer","synge","sauna","tonga","tings","promo","hides","clair","elisa","bower","reins","diann","lubed","nulls","picks","laban","milch","buber","stomp","bosom","lying","haled","avert","wries","macon","skids","fumed","ogles","clods","antic","nosey","crimp","purge","mommy","cased","taxes","covet","clack","butch","panty","lents","machs","exude","tooth","adore","shuck","asses","after","terra","dices","aryan","regor","romes","stile","cairo","maura","flail","eaves","estes","sousa","visas","baron","civet","kitty","freed","ralph","tango","gawks","cheat","study","fancy","fiber","musks","souse","brims","claim","bikes","venue","sired","thymi","rivas","skimp","pleas","woman","gimpy","cawed","minos","pints","knock","poked","bowen","risky","towel","oinks","linus","heals","pears","codas","inner","pitch","harpy","niger","madly","bumpy","stair","files","nobel","celli","spars","jades","balmy","kooky","plums","trues","gloss","trims","daunt","tubby","dared","wadis","smell","darby","stink","drill","dover","ruler","laden","dikes","layla","fells","maker","joked","horns","these","baize","spahn","whens","edged","mushy","plume","tucks","spurs","husky","dried","bigot","pupas","drily","aware","hagar","newly","knots","pratt","feces","sabik","watts","cooke","riles","seamy","fleas","dusts","barfs","roans","pawns","vivid","kirks","tania","feral","tubae","horne","aries","brits","combs","chunk","stork","waned","texan","elide","glens","emery","autos","trams","dosed","cheri","baits","jacks","whose","fazed","matte","swans","maxed","write","spays","orion","traci","horse","stars","strut","goods","verge","scuff","award","dives","wires","burnt","dimly","sleds","mayan","biped","quirk","sofia","slabs","waste","robby","mayor","fatty","items","bowel","mires","swarm","route","swash","sooth","paved","steak","upend","sough","throw","perts","stave","carry","burgs","hilts","plane","toady","nadir","stick","foist","gnarl","spain","enter","sises","story","scarf","ryder","glums","nappy","sixes","honed","marcy","offer","kneel","leeds","lites","voter","vince","bursa","heave","roses","trees","argos","leann","grimy","zelma","crick","tract","flips","folks","smote","brier","moore","goose","baden","riled","looks","sober","tusks","house","acmes","lubes","chows","neath","vivas","defer","allay","casey","kmart","pests","proms","eying","cider","leave","shush","shots","karla","scorn","gifts","sneer","mercy","copes","faxed","spurt","monet","awoke","rocky","share","gores","drawl","tears","mooed","nones","wined","wrens","modem","beria","hovel","retch","mates","hands","stymy","peace","carat","coots","hotel","karen","hayed","mamet","cuing","paper","rages","suave","reuse","auden","costs","loner","rapes","hazel","rites","brent","pumps","dutch","puffs","noons","grams","teats","cease","honda","pricy","forgo","fleck","hired","silos","merge","rafts","halon","larks","deere","jello","cunts","sifts","boner","morin","mimes","bungs","marie","harts","snobs","sonic","hippo","comes","crops","mango","wrung","garbs","natty","cents","fitch","moldy","adams","sorta","coeds","gilds","kiddy","nervy","slurp","ramon","fuzed","hiker","winks","vanes","goody","hawks","crowd","bract","marla","limbs","solve","gloom","sloop","eaton","memos","tames","heirs","berms","wanes","faint","numbs","holes","grubs","rakes","waist","miser","stays","antis","marsh","skyed","payne","champ","jimmy","clues","fatal","shoed","freon","lopez","snowy","loins","stale","thank","reads","isles","grill","align","saxes","rubin","rigel","walls","beers","wispy","topic","alden","anton","ducts","david","duets","fries","oiled","waken","allot","swats","woozy","tuxes","inter","dunne","known","axles","graph","bumps","jerry","hitch","crews","lucia","banal","grope","valid","meres","thick","lofts","chaff","taker","glues","snubs","trawl","keels","liker","stand","harps","casks","nelly","debby","panes","dumps","norma","racks","scams","forte","dwell","dudes","hypos","sissy","swamp","faust","slake","maven","lowed","lilts","bobby","gorey","swear","nests","marci","palsy","siege","oozes","rates","stunt","herod","wilma","other","girts","conic","goner","peppy","class","sized","games","snell","newsy","amend","solis","duane","troop","linda","tails","woofs","scuds","shies","patti","stunk","acres","tevet","allen","carpi","meets","trend","salty","galls","crept","toner","panda","cohen","chase","james","bravo","styed","coals","oates","swami","staph","frisk","cares","cords","stems","razed","since","mopes","rices","junes","raged","liter","manes","rearm","naive","tyree","medic","laded","pearl","inset","graft","chair","votes","saver","cains","knobs","gamay","hunch","crags","olson","teams","surge","wests","boney","limos","ploys","algae","gaols","caked","molts","glops","tarot","wheal","cysts","husks","vaunt","beaus","fauns","jeers","mitty","stuff","shape","sears","buffy","maced","fazes","vegas","stamp","borer","gaged","shade","finds","frock","plods","skied","stump","ripes","chick","cones","fixed","coled","rodeo","basil","dazes","sting","surfs","mindy","creak","swung","cadge","franc","seven","sices","weest","unite","codex","trick","fusty","plaid","hills","truck","spiel","sleek","anons","pupae","chiba","hoops","trash","noted","boris","dough","shirt","cowls","seine","spool","miens","yummy","grade","proxy","hopes","girth","deter","dowry","aorta","paean","corms","giant","shank","where","means","years","vegan","derek","tales"}))
	//fmt.Println(ladderLength("qa","sq",[]string{"si","go","se","cm","so","ph","mt","db","mb","sb","kr","ln","tm","le","av","sm","ar","ci","ca","br","ti","ba","to","ra","fa","yo","ow","sn","ya","cr","po","fe","ho","ma","re","or","rn","au","ur","rh","sr","tc","lt","lo","as","fr","nb","yb","if","pb","ge","th","pm","rb","sh","co","ga","li","ha","hz","no","bi","di","hi","qa","pi","os","uh","wm","an","me","mo","na","la","st","er","sc","ne","mn","mi","am","ex","pt","io","be","fm","ta","tb","ni","mr","pa","he","lr","sq","ye"}))
	//fmt.Println(subsetsWithDup([]int{1,2,2}))
	//fmt.Println(largestRectangleArea([]int{1,2,3,4,5,6,7,8,9}))
	//fmt.Println(largestRectangleArea([]int{1,2,3}))
	//[2,1,5,6,2,3]
	//fmt.Println(titleToNumber("AB"))
	//fmt.Println(combinationSum2([]int{10,1,2,7,6,1,5},8))
	//fmt.Println(jump([]int{8,2,4,4,4,9,5,2,5,8,8,0,8,6,9,1,1,6,3,5,1,2,6,6,0,4,8,6,0,3,2,8,7,6,5,1,7,0,3,4,8,3,5,9,0,4,0,1,0,5,9,2,0,7,0,2,1,0,8,2,5,1,2,3,9,7,4,7,0,0,1,8,5,6,7,5,1,9,9,3,5,0,7,5}))
	//fmt.Println(searchRange([]int{1},1))
	//fmt.Println(canJump([]int{3,2,1,0,4}))
	/*fmt.Println(merge([]Interval{
		Interval{2,3},
		Interval{5,6},
		Interval{1,10},
	}))*/
	//fmt.Println(rotateRight(&ListNode{},1))
	//fmt.Println(fullJustify([]string{"What","must","be","shall","be."},12))
	//fmt.Println(uniquePathsWithObstacles([][]int{[]int{0,0},[]int{1,0}}))
	//fmt.Println(minDistance("a","b"))
}

/*
"nanny"
"aloud"
["ricky","grind","cubic","panic","lover","farce","gofer","sales","flint","omens","lipid","briny","cloth","anted","slime","oaten","harsh","touts","stoop","cabal","lazed","elton","skunk","nicer","pesky","kusch","bused","kinda","tunis","enjoy","aches","prowl","babar","rooms","burst","slush","pines","urine","pinky","bayed","mania","light","flare","wares","women","verne","moron","shine","bluer","zeros","bleak","brief","tamra","vasts","jamie","lairs","penal","worst","yowls","pills","taros","addle","alyce","creep","saber","floyd","cures","soggy","vexed","vilma","cabby","verde","euler","cling","wanna","jenny","donor","stole","sakha","blake","sanes","riffs","forge","horus","sered","piked","prosy","wases","glove","onset","spake","benin","talks","sites","biers","wendy","dante","allan","haven","nears","shaka","sloth","perky","spear","spend","clint","dears","sadly","units","vista","hinds","marat","natal","least","bough","pales","boole","ditch","greys","slunk","bitch","belts","sense","skits","monty","yawns","music","hails","alien","gibes","lille","spacy","argot","wasps","drubs","poops","bella","clone","beast","emend","iring","start","darla","bells","cults","dhaka","sniff","seers","bantu","pages","fever","tacky","hoses","strop","climb","pairs","later","grant","raven","stael","drips","lucid","awing","dines","balms","della","galen","toned","snips","shady","chili","fears","nurse","joint","plump","micky","lions","jamal","queer","ruins","frats","spoof","semen","pulps","oldie","coors","rhone","papal","seals","spans","scaly","sieve","klaus","drums","tided","needs","rider","lures","treks","hares","liner","hokey","boots","primp","laval","limes","putts","fonda","damon","pikes","hobbs","specs","greet","ketch","braid","purer","tsars","berne","tarts","clean","grate","trips","chefs","timex","vicky","pares","price","every","beret","vices","jodie","fanny","mails","built","bossy","farms","pubic","gongs","magma","quads","shell","jocks","woods","waded","parka","jells","worse","diner","risks","bliss","bryan","terse","crier","incur","murky","gamed","edges","keens","bread","raced","vetch","glint","zions","porno","sizes","mends","ached","allie","bands","plank","forth","fuels","rhyme","wimpy","peels","foggy","wings","frill","edgar","slave","lotus","point","hints","germs","clung","limed","loafs","realm","myron","loopy","plush","volts","bimbo","smash","windy","sours","choke","karin","boast","whirr","tiber","dimes","basel","cutes","pinto","troll","thumb","decor","craft","tared","split","josue","tramp","screw","label","lenny","apses","slept","sikhs","child","bouts","cites","swipe","lurks","seeds","fists","hoard","steed","reams","spoil","diego","peale","bevel","flags","mazes","quart","snipe","latch","lards","acted","falls","busby","holed","mummy","wrong","wipes","carlo","leers","wails","night","pasty","eater","flunk","vedas","curse","tyros","mirth","jacky","butte","wired","fixes","tares","vague","roved","stove","swoon","scour","coked","marge","cants","comic","corns","zilch","typos","lives","truer","comma","gaily","teals","witty","hyper","croat","sways","tills","hones","dowel","llano","clefs","fores","cinch","brock","vichy","bleed","nuder","hoyle","slams","macro","arabs","tauts","eager","croak","scoop","crime","lurch","weals","fates","clipt","teens","bulls","domed","ghana","culls","frame","hanky","jared","swain","truss","drank","lobby","lumps","pansy","whews","saris","trite","weeps","dozes","jeans","flood","chimu","foxes","gelds","sects","scoff","poses","mares","famed","peers","hells","laked","zests","wring","steal","snoot","yodel","scamp","ellis","bandy","marry","jives","vises","blurb","relay","patch","haley","cubit","heine","place","touch","grain","gerry","badly","hooke","fuchs","savor","apron","judge","loren","britt","smith","tammy","altar","duels","huber","baton","dived","apace","sedan","basts","clark","mired","perch","hulks","jolly","welts","quack","spore","alums","shave","singe","lanny","dread","profs","skeet","flout","darin","newed","steer","taine","salvo","mites","rules","crash","thorn","olive","saves","yawed","pique","salon","ovens","dusty","janie","elise","carve","winds","abash","cheep","strap","fared","discs","poxed","hoots","catch","combo","maize","repay","mario","snuff","delve","cored","bards","sudan","shuns","yukon","jowls","wayne","torus","gales","creek","prove","needy","wisps","terri","ranks","books","dicky","tapes","aping","padre","roads","nines","seats","flats","rains","moira","basic","loves","pulls","tough","gills","codes","chest","teeny","jolts","woody","flame","asked","dulls","hotly","glare","mucky","spite","flake","vines","lindy","butts","froth","beeps","sills","bunny","flied","shaun","mawed","velds","voled","doily","patel","snake","thigh","adler","calks","desks","janus","spunk","baled","match","strip","hosed","nippy","wrest","whams","calfs","sleet","wives","boars","chain","table","duked","riped","edens","galas","huffs","biddy","claps","aleut","yucks","bangs","quids","glenn","evert","drunk","lusts","senna","slate","manet","roted","sleep","loxes","fluky","fence","clamp","doted","broad","sager","spark","belch","mandy","deana","beyer","hoist","leafy","levee","libel","tonic","aloes","steam","skews","tides","stall","rifts","saxon","mavis","asama","might","dotes","tangs","wroth","kited","salad","liens","clink","glows","balky","taffy","sided","sworn","oasis","tenth","blurt","tower","often","walsh","sonny","andes","slump","scans","boded","chive","finer","ponce","prune","sloes","dined","chums","dingo","harte","ahead","event","freer","heart","fetch","sated","soapy","skins","royal","cuter","loire","minot","aisle","horny","slued","panel","eight","snoop","pries","clive","pored","wrist","piped","daren","cells","parks","slugs","cubed","highs","booze","weary","stain","hoped","finny","weeds","fetid","racer","tasks","right","saint","shahs","basis","refer","chart","seize","lulls","slant","belay","clots","jinny","tours","modes","gloat","dunks","flute","conch","marts","aglow","gayer","lazes","dicks","chime","bears","sharp","hatch","forms","terry","gouda","thins","janet","tonya","axons","sewed","danny","rowdy","dolts","hurry","opine","fifty","noisy","spiky","humid","verna","poles","jayne","pecos","hooky","haney","shams","snots","sally","ruder","tempe","plunk","shaft","scows","essie","dated","fleet","spate","bunin","hikes","sodas","filly","thyme","fiefs","perks","chary","kiths","lidia","lefty","wolff","withe","three","crawl","wotan","brown","japed","tolls","taken","threw","crave","clash","layer","tends","notes","fudge","musky","bawdy","aline","matts","shirr","balks","stash","wicks","crepe","foods","fares","rotes","party","petty","press","dolly","mangy","leeks","silly","leant","nooks","chapt","loose","caged","wages","grist","alert","sheri","moody","tamps","hefts","souls","rubes","rolex","skulk","veeps","nonce","state","level","whirl","bight","grits","reset","faked","spiny","mixes","hunks","major","missy","arius","damns","fitly","caped","mucus","trace","surat","lloyd","furry","colin","texts","livia","reply","twill","ships","peons","shear","norms","jumbo","bring","masks","zippy","brine","dorks","roded","sinks","river","wolfs","strew","myths","pulpy","prank","veins","flues","minus","phone","banns","spell","burro","brags","boyle","lambs","sides","knees","clews","aired","skirt","heavy","dimer","bombs","scums","hayes","chaps","snugs","dusky","loxed","ellen","while","swank","track","minim","wiled","hazed","roofs","cantu","sorry","roach","loser","brass","stint","jerks","dirks","emory","campy","poise","sexed","gamer","catty","comte","bilbo","fasts","ledge","drier","idles","doors","waged","rizal","pured","weirs","crisp","tasty","sored","palmy","parts","ethel","unify","crows","crest","udder","delis","punks","dowse","totes","emile","coded","shops","poppa","pours","gushy","tiffs","shads","birds","coils","areas","boons","hulls","alter","lobes","pleat","depth","fires","pones","serra","sweat","kline","malay","ruled","calve","tired","drabs","tubed","wryer","slung","union","sonya","aided","hewed","dicey","grids","nixed","whits","mills","buffs","yucky","drops","ready","yuppy","tweet","napes","cadre","teach","rasps","dowdy","hoary","canto","posed","dumbo","kooks","reese","snaky","binge","byron","phony","safer","friar","novel","scale","huron","adorn","carla","fauna","myers","hobby","purse","flesh","smock","along","boils","pails","times","panza","lodge","clubs","colby","great","thing","peaks","diana","vance","whets","bergs","sling","spade","soaks","beach","traps","aspen","romps","boxed","fakir","weave","nerds","swazi","dotty","curls","diver","jonas","waite","verbs","yeast","lapel","barth","soars","hooks","taxed","slews","gouge","slags","chang","chafe","saved","josie","syncs","fonds","anion","actor","seems","pyrex","isiah","glued","groin","goren","waxes","tonia","whine","scads","knelt","teaks","satan","tromp","spats","merry","wordy","stake","gland","canal","donna","lends","filed","sacks","shied","moors","paths","older","pooch","balsa","riced","facet","decaf","attic","elder","akron","chomp","chump","picky","money","sheer","bolls","crabs","dorms","water","veers","tease","dummy","dumbs","lethe","halls","rifer","demon","fucks","whips","plops","fuses","focal","taces","snout","edict","flush","burps","dawes","lorry","spews","sprat","click","deann","sited","aunts","quips","godly","pupil","nanny","funks","shoon","aimed","stacy","helms","mints","banks","pinch","local","twine","pacts","deers","halos","slink","preys","potty","ruffs","pusan","suits","finks","slash","prods","dense","edsel","heeds","palls","slats","snits","mower","rares","ailed","rouge","ellie","gated","lyons","duded","links","oaths","letha","kicks","firms","gravy","month","kongo","mused","ducal","toted","vocal","disks","spied","studs","macao","erick","coupe","starr","reaps","decoy","rayon","nicks","breed","cosby","haunt","typed","plain","trays","muled","saith","drano","cower","snows","buses","jewry","argus","doers","flays","swish","resin","boobs","sicks","spies","bails","wowed","mabel","check","vapid","bacon","wilda","ollie","loony","irked","fraud","doles","facts","lists","gazed","furls","sunks","stows","wilde","brick","bowed","guise","suing","gates","niter","heros","hyped","clomp","never","lolls","rangy","paddy","chant","casts","terns","tunas","poker","scary","maims","saran","devon","tripe","lingo","paler","coped","bride","voted","dodge","gross","curds","sames","those","tithe","steep","flaks","close","swops","stare","notch","prays","roles","crush","feuds","nudge","baned","brake","plans","weepy","dazed","jenna","weiss","tomes","stews","whist","gibed","death","clank","cover","peeks","quick","abler","daddy","calls","scald","lilia","flask","cheer","grabs","megan","canes","jules","blots","mossy","begun","freak","caved","hello","hades","theed","wards","darcy","malta","peter","whorl","break","downs","odder","hoofs","kiddo","macho","fords","liked","flees","swing","elect","hoods","pluck","brook","astir","bland","sward","modal","flown","ahmad","waled","craps","cools","roods","hided","plath","kings","grips","gives","gnats","tabby","gauls","think","bully","fogey","sawed","lints","pushy","banes","drake","trail","moral","daley","balds","chugs","geeky","darts","soddy","haves","opens","rends","buggy","moles","freud","gored","shock","angus","puree","raves","johns","armed","packs","minis","reich","slots","totem","clown","popes","brute","hedge","latin","stoke","blend","pease","rubik","greer","hindi","betsy","flows","funky","kelli","humps","chewy","welds","scowl","yells","cough","sasha","sheaf","jokes","coast","words","irate","hales","camry","spits","burma","rhine","bends","spill","stubs","power","voles","learn","knoll","style","twila","drove","dacca","sheen","papas","shale","jones","duped","tunny","mouse","floss","corks","skims","swaps","inned","boxer","synch","skies","strep","bucks","belau","lower","flaky","quill","aural","rufus","floes","pokes","sends","sates","dally","boyer","hurts","foyer","gowns","torch","luria","fangs","moats","heinz","bolts","filet","firth","begot","argue","youth","chimp","frogs","kraft","smite","loges","loons","spine","domes","pokey","timur","noddy","doggy","wades","lanes","hence","louts","turks","lurid","goths","moist","bated","giles","stood","winos","shins","potts","brant","vised","alice","rosie","dents","babes","softy","decay","meats","tanya","rusks","pasts","karat","nuked","gorge","kinks","skull","noyce","aimee","watch","cleat","stuck","china","testy","doses","safes","stage","bayes","twins","limps","denis","chars","flaps","paces","abase","grays","deans","maria","asset","smuts","serbs","whigs","vases","robyn","girls","pents","alike","nodal","molly","swigs","swill","slums","rajah","bleep","beget","thanh","finns","clock","wafts","wafer","spicy","sorer","reach","beats","baker","crown","drugs","daisy","mocks","scots","fests","newer","agate","drift","marta","chino","flirt","homed","bribe","scram","bulks","servo","vesta","divas","preps","naval","tally","shove","ragas","blown","droll","tryst","lucky","leech","lines","sires","pyxed","taper","trump","payee","midge","paris","bored","loads","shuts","lived","swath","snare","boned","scars","aeons","grime","writs","paige","rungs","blent","signs","davis","dials","daubs","rainy","fawns","wrier","golds","wrath","ducks","allow","hosea","spike","meals","haber","muses","timed","broom","burks","louis","gangs","pools","vales","altai","elope","plied","slain","chasm","entry","slide","bawls","title","sings","grief","viola","doyle","peach","davit","bench","devil","latex","miles","pasha","tokes","coves","wheel","tried","verdi","wanda","sivan","prior","fryer","plots","kicky","porch","shill","coats","borne","brink","pawed","erwin","tense","stirs","wends","waxen","carts","smear","rival","scare","phase","bragg","crane","hocks","conan","bests","dares","molls","roots","dunes","slips","waked","fours","bolds","slosh","yemen","poole","solid","ports","fades","legal","cedes","green","curie","seedy","riper","poled","glade","hosts","tools","razes","tarry","muddy","shims","sword","thine","lasts","bloat","soled","tardy","foots","skiff","volta","murks","croci","gooks","gamey","pyxes","poems","kayla","larva","slaps","abuse","pings","plows","geese","minks","derby","super","inked","manic","leaks","flops","lajos","fuzes","swabs","twigs","gummy","pyres","shrew","islet","doled","wooly","lefts","hunts","toast","faith","macaw","sonia","leafs","colas","conks","altos","wiped","scene","boors","patsy","meany","chung","wakes","clear","ropes","tahoe","zones","crate","tombs","nouns","garth","puked","chats","hanks","baked","binds","fully","soaps","newel","yarns","puers","carps","spelt","lully","towed","scabs","prime","blest","patty","silky","abner","temps","lakes","tests","alias","mines","chips","funds","caret","splat","perry","turds","junks","cramp","saned","peary","snarl","fired","stung","nancy","bulge","styli","seams","hived","feast","triad","jaded","elvin","canny","birth","routs","rimed","pusey","laces","taste","basie","malls","shout","prier","prone","finis","claus","loops","heron","frump","spare","menus","ariel","crams","bloom","foxed","moons","mince","mixed","piers","deres","tempt","dryer","atone","heats","dario","hawed","swims","sheet","tasha","dings","clare","aging","daffy","wried","foals","lunar","havel","irony","ronny","naves","selma","gurus","crust","percy","murat","mauro","cowed","clang","biker","harms","barry","thump","crude","ulnae","thong","pager","oases","mered","locke","merle","soave","petal","poser","store","winch","wedge","inlet","nerdy","utter","filth","spray","drape","pukes","ewers","kinds","dates","meier","tammi","spoor","curly","chill","loped","gooey","boles","genet","boost","beets","heath","feeds","growl","livid","midst","rinds","fresh","waxed","yearn","keeps","rimes","naked","flick","plies","deeps","dirty","hefty","messy","hairy","walks","leper","sykes","nerve","rover","jived","brisk","lenin","viper","chuck","sinus","luger","ricks","hying","rusty","kathy","herds","wider","getty","roman","sandy","pends","fezes","trios","bites","pants","bless","diced","earth","shack","hinge","melds","jonah","chose","liver","salts","ratty","ashed","wacky","yokes","wanly","bruce","vowel","black","grail","lungs","arise","gluts","gluey","navel","coyer","ramps","miter","aldan","booth","musty","rills","darns","tined","straw","kerri","hared","lucks","metes","penny","radon","palms","deeds","earls","shard","pried","tampa","blank","gybes","vicki","drool","groom","curer","cubes","riggs","lanky","tuber","caves","acing","golly","hodge","beard","ginny","jibed","fumes","astor","quito","cargo","randi","gawky","zings","blind","dhoti","sneak","fatah","fixer","lapps","cline","grimm","fakes","maine","erika","dealt","mitch","olden","joist","gents","likes","shelf","silts","goats","leads","marin","spire","louie","evans","amuse","belly","nails","snead","model","whats","shari","quote","tacks","nutty","lames","caste","hexes","cooks","miner","shawn","anise","drama","trike","prate","ayers","loans","botch","vests","cilia","ridge","thugs","outed","jails","moped","plead","tunes","nosed","wills","lager","lacks","cried","wince","berle","flaws","boise","tibet","bided","shred","cocky","brice","delta","congo","holly","hicks","wraps","cocks","aisha","heard","cured","sades","horsy","umped","trice","dorky","curve","ferry","haler","ninth","pasta","jason","honer","kevin","males","fowls","awake","pores","meter","skate","drink","pussy","soups","bases","noyes","torts","bogus","still","soupy","dance","worry","eldon","stern","menes","dolls","dumpy","gaunt","grove","coops","mules","berry","sower","roams","brawl","greed","stags","blurs","swift","treed","taney","shame","easel","moves","leger","ville","order","spock","nifty","brian","elias","idler","serve","ashen","bizet","gilts","spook","eaten","pumas","cotes","broke","toxin","groan","laths","joins","spots","hated","tokay","elite","rawer","fiats","cards","sassy","milks","roost","glean","lutes","chins","drown","marks","pined","grace","fifth","lodes","rusts","terms","maxes","savvy","choir","savoy","spoon","halve","chord","hulas","sarah","celia","deems","ninny","wines","boggy","birch","raved","wales","beams","vibes","riots","warty","nigel","askew","faxes","sedge","sheol","pucks","cynic","relax","boers","whims","bents","candy","luann","slogs","bonny","barns","iambs","fused","duffy","guilt","bruin","pawls","penis","poppy","owing","tribe","tuner","moray","timid","ceded","geeks","kites","curio","puffy","perot","caddy","peeve","cause","dills","gavel","manse","joker","lynch","crank","golda","waits","wises","hasty","paves","grown","reedy","crypt","tonne","jerky","axing","swept","posse","rings","staff","tansy","pared","glaze","grebe","gonna","shark","jumps","vials","unset","hires","tying","lured","motes","linen","locks","mamas","nasty","mamie","clout","nader","velma","abate","tight","dales","serer","rives","bales","loamy","warps","plato","hooch","togae","damps","ofter","plumb","fifes","filmy","wiper","chess","lousy","sails","brahe","ounce","flits","hindu","manly","beaux","mimed","liken","forts","jambs","peeps","lelia","brews","handy","lusty","brads","marne","pesos","earle","arson","scout","showy","chile","sumps","hiked","crook","herbs","silks","alamo","mores","dunce","blaze","stank","haste","howls","trots","creon","lisle","pause","hates","mulch","mined","moder","devin","types","cindy","beech","tuned","mowed","pitts","chaos","colds","bidet","tines","sighs","slimy","brain","belle","leery","morse","ruben","prows","frown","disco","regal","oaken","sheds","hives","corny","baser","fated","throe","revel","bores","waved","shits","elvia","ferns","maids","color","coifs","cohan","draft","hmong","alton","stine","cluck","nodes","emily","brave","blair","blued","dress","bunts","holst","clogs","rally","knack","demos","brady","blues","flash","goofy","blocs","diane","colic","smile","yules","foamy","splay","bilge","faker","foils","condo","knell","crack","gallo","purls","auras","cakes","doves","joust","aides","lades","muggy","tanks","middy","tarps","slack","capet","frays","donny","venal","yeats","misty","denim","glass","nudes","seeps","gibbs","blows","bobbi","shane","yards","pimps","clued","quiet","witch","boxes","prawn","kerry","torah","kinko","dingy","emote","honor","jelly","grins","trope","vined","bagel","arden","rapid","paged","loved","agape","mural","budge","ticks","suers","wendi","slice","salve","robin","bleat","batik","myles","teddy","flatt","puppy","gelid","largo","attar","polls","glide","serum","fundy","sucks","shalt","sewer","wreak","dames","fonts","toxic","hines","wormy","grass","louse","bowls","crass","benny","moire","margo","golfs","smart","roxie","wight","reign","dairy","clops","paled","oddly","sappy","flair","shown","bulgy","benet","larch","curry","gulfs","fends","lunch","dukes","doris","spoke","coins","manna","conga","jinns","eases","dunno","tisha","swore","rhino","calms","irvin","clans","gully","liege","mains","besot","serge","being","welch","wombs","draco","lynda","forty","mumps","bloch","ogden","knits","fussy","alder","danes","loyal","valet","wooer","quire","liefs","shana","toyed","forks","gages","slims","cloys","yates","rails","sheep","nacho","divan","honks","stone","snack","added","basal","hasps","focus","alone","laxes","arose","lamed","wrapt","frail","clams","plait","hover","tacos","mooch","fault","teeth","marva","mucks","tread","waves","purim","boron","horde","smack","bongo","monte","swirl","deals","mikes","scold","muter","sties","lawns","fluke","jilts","meuse","fives","sulky","molds","snore","timmy","ditty","gasps","kills","carey","jawed","byers","tommy","homer","hexed","dumas","given","mewls","smelt","weird","speck","merck","keats","draws","trent","agave","wells","chews","blabs","roves","grieg","evens","alive","mulls","cared","garbo","fined","happy","trued","rodes","thurs","cadet","alvin","busch","moths","guild","staci","lever","widen","props","hussy","lamer","riley","bauer","chirp","rants","poxes","shyer","pelts","funny","slits","tinge","ramos","shift","caper","credo","renal","veils","covey","elmer","mated","tykes","wooed","briar","gears","foley","shoes","decry","hypes","dells","wilds","runts","wilts","white","easts","comer","sammy","lochs","favor","lance","dawns","bushy","muted","elsie","creel","pocks","tenet","cagey","rides","socks","ogled","soils","sofas","janna","exile","barks","frank","takes","zooms","hakes","sagan","scull","heaps","augur","pouch","blare","bulbs","wryly","homey","tubas","limbo","hardy","hoagy","minds","bared","gabby","bilks","float","limns","clasp","laura","range","brush","tummy","kilts","cooed","worms","leary","feats","robes","suite","veals","bosch","moans","dozen","rarer","slyer","cabin","craze","sweet","talon","treat","yanks","react","creed","eliza","sluts","cruet","hafts","noise","seder","flies","weeks","venus","backs","eider","uriel","vouch","robed","hacks","perth","shiny","stilt","torte","throb","merer","twits","reeds","shawl","clara","slurs","mixer","newts","fried","woolf","swoop","kaaba","oozed","mayer","caned","laius","lunge","chits","kenny","lifts","mafia","sowed","piled","stein","whack","colts","warms","cleft","girds","seeks","poets","angel","trade","parsi","tiers","rojas","vexes","bryce","moots","grunt","drain","lumpy","stabs","poohs","leapt","polly","cuffs","giddy","towns","dacha","quoth","provo","dilly","carly","mewed","tzars","crock","toked","speak","mayas","pssts","ocher","motel","vogue","camps","tharp","taunt","drone","taint","badge","scott","scats","bakes","antes","gruel","snort","capes","plate","folly","adobe","yours","papaw","hench","moods","clunk","chevy","tomas","narcs","vonda","wiles","prigs","chock","laser","viced","stiff","rouse","helps","knead","gazer","blade","tumid","avail","anger","egged","guide","goads","rabin","toddy","gulps","flank","brats","pedal","junky","marco","tinny","tires","flier","satin","darth","paley","gumbo","rared","muffs","rower","prude","frees","quays","homes","munch","beefs","leash","aston","colon","finch","bogey","leaps","tempo","posts","lined","gapes","locus","maori","nixes","liven","songs","opted","babel","wader","barer","farts","lisps","koran","lathe","trill","smirk","mamma","viler","scurf","ravel","brigs","cooky","sachs","fulls","goals","turfs","norse","hauls","cores","fairy","pluto","kneed","cheek","pangs","risen","czars","milne","cribs","genes","wefts","vents","sages","seres","owens","wiley","flume","haded","auger","tatty","onion","cater","wolfe","magic","bodes","gulls","gazes","dandy","snags","rowed","quell","spurn","shore","veldt","turns","slavs","coach","stalk","snuck","piles","orate","joyed","daily","crone","wager","solos","earns","stark","lauds","kasey","villa","gnaws","scent","wears","fains","laced","tamer","pipes","plant","lorie","rivet","tamed","cozen","theme","lifer","sunny","shags","flack","gassy","eased","jeeps","shire","fargo","timer","brash","behan","basin","volga","krone","swiss","docks","booed","ebert","gusty","delay","oared","grady","buick","curbs","crete","lucas","strum","besom","gorse","troth","donne","chink","faced","ahmed","texas","longs","aloud","bethe","cacao","hilda","eagle","karyn","harks","adder","verse","drays","cello","taped","snide","taxis","kinky","penes","wicca","sonja","aways","dyers","bolas","elfin","slope","lamps","hutch","lobed","baaed","masts","ashes","ionic","joyce","payed","brays","malts","dregs","leaky","runny","fecal","woven","hurls","jorge","henna","dolby","booty","brett","dykes","rural","fight","feels","flogs","brunt","preen","elvis","dopey","gripe","garry","gamma","fling","space","mange","storm","arron","hairs","rogue","repel","elgar","ruddy","cross","medan","loses","howdy","foams","piker","halts","jewel","avery","stool","cruel","cases","ruses","cathy","harem","flour","meted","faces","hobos","charm","jamar","cameo","crape","hooey","reefs","denny","mitts","sores","smoky","nopes","sooty","twirl","toads","vader","julep","licks","arias","wrote","north","bunks","heady","batch","snaps","claws","fouls","faded","beans","wimps","idled","pulse","goons","noose","vowed","ronda","rajas","roast","allah","punic","slows","hours","metal","slier","meaty","hanna","curvy","mussy","truth","troys","block","reels","print","miffs","busts","bytes","cream","otter","grads","siren","kilos","dross","batty","debts","sully","bares","baggy","hippy","berth","gorky","argon","wacko","harry","smoke","fails","perms","score","steps","unity","couch","kelly","rumps","fines","mouth","broth","knows","becky","quits","lauri","trust","grows","logos","apter","burrs","zincs","buyer","bayer","moose","overt","croon","ousts","lands","lithe","poach","jamel","waive","wiser","surly","works","paine","medal","glads","gybed","paint","lorre","meant","smugs","bryon","jinni","sever","viols","flubs","melts","heads","peals","aiken","named","teary","yalta","styes","heist","bongs","slops","pouts","grape","belie","cloak","rocks","scone","lydia","goofs","rents","drive","crony","orlon","narks","plays","blips","pence","march","alger","baste","acorn","billy","croce","boone","aaron","slobs","idyls","irwin","elves","stoat","doing","globe","verve","icons","trial","olsen","pecks","there","blame","tilde","milky","sells","tangy","wrack","fills","lofty","truce","quark","delia","stowe","marty","overs","putty","coral","swine","stats","swags","weans","spout","bulky","farsi","brest","gleam","beaks","coons","hater","peony","huffy","exert","clips","riven","payer","doped","salas","meyer","dryad","thuds","tilts","quilt","jetty","brood","gulch","corps","tunic","hubby","slang","wreck","purrs","punch","drags","chide","sulks","tints","huger","roped","dopes","booby","rosin","outer","gusto","tents","elude","brows","lease","ceres","laxer","worth","necks","races","corey","trait","stuns","soles","teems","scrip","privy","sight","minor","alisa","stray","spank","cress","nukes","rises","gusts","aurae","karma","icing","prose","biked","grand","grasp","skein","shaky","clump","rummy","stock","twain","zoned","offed","ghats","mover","randy","vault","craws","thees","salem","downy","sangs","chore","cited","grave","spinx","erica","raspy","dying","skips","clerk","paste","moved","rooks","intel","moses","avers","staid","yawls","blast","lyres","monks","gaits","floor","saner","waver","assam","infer","wands","bunch","dryly","weedy","honey","baths","leach","shorn","shows","dream","value","dooms","spiro","raped","shook","stead","moran","ditto","loots","tapir","looms","clove","stops","pinks","soppy","ripen","wench","shone","bauds","doric","leans","nadia","cries","camus","boozy","maris","fools","morns","bides","greek","gauss","roget","lamar","hazes","beefy","dupes","refed","felts","larry","guile","ables","wants","warns","toils","bathe","edger","paced","rinks","shoos","erich","whore","tiger","jumpy","lamas","stack","among","punts","scalp","alloy","solon","quite","comas","whole","parse","tries","reeve","tiled","deena","roomy","rodin","aster","twice","musts","globs","parch","drawn","filch","bonds","tells","droop","janis","holds","scant","lopes","based","keven","whiny","aspic","gains","franz","jerri","steel","rowel","vends","yelps","begin","logic","tress","sunni","going","barge","blood","burns","basks","waifs","bones","skill","hewer","burly","clime","eking","withs","capek","berta","cheap","films","scoot","tweed","sizer","wheat","acton","flung","ponds","tracy","fiver","berra","roger","mutes","burke","miked","valve","whisk","runes","parry","toots","japes","roars","rough","irons","romeo","cages","reeks","cigar","saiph","dully","hangs","chops","rolls","prick","acuff","spent","sulla","train","swell","frets","names","anita","crazy","sixth","blunt","fewer","large","brand","slick","spitz","rears","ogres","toffy","yolks","flock","gnawn","eries","blink","skier","feted","tones","snail","ether","barbs","noses","hears","upset","awash","cloud","trunk","degas","dungs","rated","shall","yeahs","coven","sands","susan","fable","gunny","began","serfs","balls","dinky","madge","prong","spilt","lilly","brawn","comet","spins","raids","dries","sorts","makes","mason","mayra","royce","stout","mealy","pagan","nasal","folds","libby","coups","photo","mosey","amens","speed","lords","board","fetal","lagos","scope","raked","bonus","mutts","willy","sport","bingo","thant","araby","bette","rebel","gases","small","humus","grosz","beset","slays","steve","scrap","blahs","south","pride","heels","tubes","beady","lacey","genus","mauls","vying","spice","sexes","ester","drams","today","comae","under","jests","direr","yoked","tempi","early","boats","jesus","warts","guppy","gilda","quota","token","edwin","ringo","gaped","lemon","hurst","manor","arrow","mists","prize","silas","blobs","diets","ervin","stony","buddy","bates","rabid","ducat","ewing","jaunt","beads","doyen","blush","thoth","tiles","piper","short","peron","alley","decks","shunt","whirs","cushy","roils","betty","plugs","woken","jibes","foray","merak","ruing","becks","whale","shoot","dwelt","spawn","fairs","dozed","celts","blond","tikes","sabin","feint","vamps","cokes","willa","slues","bills","force","curst","yokel","surer","miler","fices","arced","douse","hilly","lucio","tongs","togas","minty","sagas","pates","welsh","bruno","decal","elate","linux","gyros","pryor","mousy","pains","shake","spica","pupal","probe","mount","shirk","purus","kilns","rests","graze","hague","spuds","sweep","momma","burch","maces","samar","brace","riser","booms","build","camel","flyer","synge","sauna","tonga","tings","promo","hides","clair","elisa","bower","reins","diann","lubed","nulls","picks","laban","milch","buber","stomp","bosom","lying","haled","avert","wries","macon","skids","fumed","ogles","clods","antic","nosey","crimp","purge","mommy","cased","taxes","covet","clack","butch","panty","lents","machs","exude","tooth","adore","shuck","asses","after","terra","dices","aryan","regor","romes","stile","cairo","maura","flail","eaves","estes","sousa","visas","baron","civet","kitty","freed","ralph","tango","gawks","cheat","study","fancy","fiber","musks","souse","brims","claim","bikes","venue","sired","thymi","rivas","skimp","pleas","woman","gimpy","cawed","minos","pints","knock","poked","bowen","risky","towel","oinks","linus","heals","pears","codas","inner","pitch","harpy","niger","madly","bumpy","stair","files","nobel","celli","spars","jades","balmy","kooky","plums","trues","gloss","trims","daunt","tubby","dared","wadis","smell","darby","stink","drill","dover","ruler","laden","dikes","layla","fells","maker","joked","horns","these","baize","spahn","whens","edged","mushy","plume","tucks","spurs","husky","dried","bigot","pupas","drily","aware","hagar","newly","knots","pratt","feces","sabik","watts","cooke","riles","seamy","fleas","dusts","barfs","roans","pawns","vivid","kirks","tania","feral","tubae","horne","aries","brits","combs","chunk","stork","waned","texan","elide","glens","emery","autos","trams","dosed","cheri","baits","jacks","whose","fazed","matte","swans","maxed","write","spays","orion","traci","horse","stars","strut","goods","verge","scuff","award","dives","wires","burnt","dimly","sleds","mayan","biped","quirk","sofia","slabs","waste","robby","mayor","fatty","items","bowel","mires","swarm","route","swash","sooth","paved","steak","upend","sough","throw","perts","stave","carry","burgs","hilts","plane","toady","nadir","stick","foist","gnarl","spain","enter","sises","story","scarf","ryder","glums","nappy","sixes","honed","marcy","offer","kneel","leeds","lites","voter","vince","bursa","heave","roses","trees","argos","leann","grimy","zelma","crick","tract","flips","folks","smote","brier","moore","goose","baden","riled","looks","sober","tusks","house","acmes","lubes","chows","neath","vivas","defer","allay","casey","kmart","pests","proms","eying","cider","leave","shush","shots","karla","scorn","gifts","sneer","mercy","copes","faxed","spurt","monet","awoke","rocky","share","gores","drawl","tears","mooed","nones","wined","wrens","modem","beria","hovel","retch","mates","hands","stymy","peace","carat","coots","hotel","karen","hayed","mamet","cuing","paper","rages","suave","reuse","auden","costs","loner","rapes","hazel","rites","brent","pumps","dutch","puffs","noons","grams","teats","cease","honda","pricy","forgo","fleck","hired","silos","merge","rafts","halon","larks","deere","jello","cunts","sifts","boner","morin","mimes","bungs","marie","harts","snobs","sonic","hippo","comes","crops","mango","wrung","garbs","natty","cents","fitch","moldy","adams","sorta","coeds","gilds","kiddy","nervy","slurp","ramon","fuzed","hiker","winks","vanes","goody","hawks","crowd","bract","marla","limbs","solve","gloom","sloop","eaton","memos","tames","heirs","berms","wanes","faint","numbs","holes","grubs","rakes","waist","miser","stays","antis","marsh","skyed","payne","champ","jimmy","clues","fatal","shoed","freon","lopez","snowy","loins","stale","thank","reads","isles","grill","align","saxes","rubin","rigel","walls","beers","wispy","topic","alden","anton","ducts","david","duets","fries","oiled","waken","allot","swats","woozy","tuxes","inter","dunne","known","axles","graph","bumps","jerry","hitch","crews","lucia","banal","grope","valid","meres","thick","lofts","chaff","taker","glues","snubs","trawl","keels","liker","stand","harps","casks","nelly","debby","panes","dumps","norma","racks","scams","forte","dwell","dudes","hypos","sissy","swamp","faust","slake","maven","lowed","lilts","bobby","gorey","swear","nests","marci","palsy","siege","oozes","rates","stunt","herod","wilma","other","girts","conic","goner","peppy","class","sized","games","snell","newsy","amend","solis","duane","troop","linda","tails","woofs","scuds","shies","patti","stunk","acres","tevet","allen","carpi","meets","trend","salty","galls","crept","toner","panda","cohen","chase","james","bravo","styed","coals","oates","swami","staph","frisk","cares","cords","stems","razed","since","mopes","rices","junes","raged","liter","manes","rearm","naive","tyree","medic","laded","pearl","inset","graft","chair","votes","saver","cains","knobs","gamay","hunch","crags","olson","teams","surge","wests","boney","limos","ploys","algae","gaols","caked","molts","glops","tarot","wheal","cysts","husks","vaunt","beaus","fauns","jeers","mitty","stuff","shape","sears","buffy","maced","fazes","vegas","stamp","borer","gaged","shade","finds","frock","plods","skied","stump","ripes","chick","cones","fixed","coled","rodeo","basil","dazes","sting","surfs","mindy","creak","swung","cadge","franc","seven","sices","weest","unite","codex","trick","fusty","plaid","hills","truck","spiel","sleek","anons","pupae","chiba","hoops","trash","noted","boris","dough","shirt","cowls","seine","spool","miens","yummy","grade","proxy","hopes","girth","deter","dowry","aorta","paean","corms","giant","shank","where","means","years","vegan","derek","tales"]
*/
