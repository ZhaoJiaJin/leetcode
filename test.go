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

func solve1(board [][]byte) bool {
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] == '.' {
				for c := '1'; c <= '9'; c++ {
					if validSk(board, i, j, byte(c)) {
						board[i][j] = byte(c)

						if solve1(board) {
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
	solve1(board)
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


func longestConsecutive(nums []int) int {
	if len(nums) <= 1{
		return len(nums)
	}
	sort(nums)    
	fmt.Println(nums)
	l := 1
	max := 1
	for i:=1;i < len(nums); i++{
		if nums[i] == nums[i-1]{
			continue
		}
		if nums[i] == nums[i-1]+1{
			l ++
		}else{
			if l > max{
				max = l
			}
			l = 1
		}
	
	}
	if l > max{
		max = l
	}
	return max
}

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
var total int 
func sumNumbers(root *TreeNode) int {
	total = 0
	if root == nil{
		return total
	}
	tmp := 0
	snhelper(root,tmp)
	return total
}

func snhelper(root *TreeNode,tmp int){
	if root == nil{
		return
	}
	if root.Left == nil && root.Right == nil{
		tmp = tmp*10 + root.Val
		total += tmp
		return
	}
	tmp = tmp*10 + root.Val
	snhelper(root.Left,tmp)
	snhelper(root.Right,tmp)
}

type P struct{
	rn int
	cn int
}

type StackP []P

func (s *StackP)pop()(P){
	ans := (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return ans
}

func (s *StackP)push(p P){
	*s = append(*s,p)
}

func (s *StackP)empty()(bool){
	return len(*s) == 0
}

func mkboard(m,n int)[][]bool{
	ret := make([][]bool,m)
	for i := 0; i < m; i++{
		ret[i] = make([]bool,n)
	}
	return ret
}


func findfree(board [][]byte,visited [][]bool,row,col int){
	maxrow := len(board)
	maxcol := len(board[0])
	var q StackP
	q.push(P{row,col})
	visited[row][col] = true
	for !q.empty(){
		now := q.pop()
		//up
		if now.rn - 1 >= 0 && board[now.rn-1][now.cn] == 'O' && visited[now.rn-1][now.cn] == false{
			up_p := P{now.rn-1,now.cn}
			q.push(up_p)
			visited[up_p.rn][up_p.cn] = true
		}
		//down
		if now.rn + 1 < maxrow && board[now.rn+1][now.cn] == 'O' && visited[now.rn+1][now.cn] == false{
			up_p := P{now.rn+1,now.cn}
			q.push(up_p)
			visited[up_p.rn][up_p.cn] = true
		}
		//left
		if now.cn - 1 >= 0 && board[now.rn][now.cn-1] == 'O' && visited[now.rn][now.cn-1] == false{
			up_p := P{now.rn,now.cn-1}
			q.push(up_p)
			visited[up_p.rn][up_p.cn] = true
		}
		//right
		if now.cn + 1 < maxcol && board[now.rn][now.cn+1] == 'O' && visited[now.rn][now.cn+1] == false{
			up_p := P{now.rn,now.cn+1}
			q.push(up_p)
			visited[up_p.rn][up_p.cn] = true
		}
	}
}


func solve(board [][]byte)  {
	row := len(board)
	if row <= 1{
		return
	}
	col := len(board[0])
	visited := mkboard(row,col)
	//first row
	for i:=0 ; i < col; i++{
		if board[0][i] == 'O' && visited[0][i] == false{
			findfree(board,visited,0,i)
		}
	}
	//last row
	for i:=0 ; i < col; i++{
		if board[row-1][i] == 'O' && visited[row-1][i] == false{
			findfree(board,visited,row-1,i)
		}
	}
	//first col
	for i:=0 ; i < row; i++{
		if board[i][0] == 'O' && visited[i][0] == false{
			findfree(board,visited,i,0)
		}
	}
	//last col
	for i:=0 ; i < row; i++{
		if board[i][col-1] == 'O' && visited[i][col-1] == false{
			findfree(board,visited,i,col-1)
		}
	}

	for i:=0 ; i < row; i++{
		for j:=0 ; j < col; j++{
			if visited[i][j] == false{
				board[i][j] = 'X'
			}
		}
	}
}

 func cancom(start int, gas, cost []int)(int,bool){
     fmt.Println("start",start)
     
     left := 0
     for i:=0; i < len(gas); i++{
         pos := (start + i) % len(gas)
         left += gas[pos]
         fmt.Println("add gas to",left)
         if cost[pos] > left{
             return start + i,false
         }
         left -= cost[pos]
         fmt.Println("left gas to",left)
     }
     return start,left >= 0
 }


 func canCompleteCircuit(gas []int, cost []int) int {
     for start:=0; start < len(gas); start ++{
         news,canc := cancom(start,gas,cost)
         if canc{
             return start
         }
         start = news
         fmt.Println("new start",start)
     }
     return -1
 }



func candy(ratings []int) int {
	size := len(ratings)
	candies := make([]int,size)
	for i := 0; i < size ; i ++{
		candies[i] = 1
	}
	for i := 1; i < size; i ++{
		if ratings[i] > ratings[i-1]{
			candies[i] = candies[i-1] + 1
		}
	}
	sum := candies[size-1]
	for i := size-2; i >= 0 ;i --{
		if ratings[i] > ratings[i+1]{
			candies[i] = max(candies[i], candies[i+1]+1)
		}
		sum += candies[i]
	}
	return sum
}


func singleNumber(nums []int) int {
	sum := 0    
	for _,v := range nums{
		sum += v
	}
	nodupsum := nums[0]
	for i:=1; i < len(nums); i ++{
		if nums[i] == nums[i-1]{
			continue
		}
		nodupsum += nums[i]
	}
	fmt.Println(sum,nodupsum)
	return (nodupsum * 3 - sum)/2
}

func isp(s string, l []string)bool{
	for _,v := range l{
		if v == s{
			return true
		}
	}
	return false
}

/*func wordBreak(s string, wordDict []string) bool {
	size := len(s)
	dp := make([]bool,size+1)
	dp[0] = true
	for i := 1 ; i <= size; i ++{
		for j:=0; j <=i ; j ++{
			if dp[j] && isp(s[j:i],wordDict){
				dp[i] = true
				break
			}
		}
		fmt.Println(dp)
	
	}
	return dp[size]
}*/

func wordBreak(s string, wordDict []string) []string {
	m := make(map[string][]string)
	return wbhlp(s,wordDict,m)
}

func wbhlp(s string, wordDict []string,m map[string][]string)[]string{
	v,p := m[s]
	if p{
		fmt.Println(s)
		return v
	}
	ret := make([]string,0)
	if s == ""{
		ret = append(ret,"")
		return ret
	}

	for _,v := range wordDict{
		if len(v) <= len(s) && v == s[:len(v)]{
			sub := wbhlp(s[len(v):],wordDict,m)
			for _,su := range sub{
				t := ""
				if su == ""{
					t = v	
				}else{
					t = v + " " + su
				}
				ret = append(ret,t)
			}
		}
	}
	m[s] = ret
	return ret
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reorderList(head *ListNode)  {
	if head == nil || head.Next == nil{
		return 
	}
	p1,p2 := head,head
	for p2.Next != nil && p2.Next.Next != nil{
		p1 = p1.Next
		p2 = p2.Next.Next
	}

	preMiddle := p1
	preCurrent := p1.Next
	for preCurrent.Next != nil{
		current := preCurrent.Next
		preCurrent.Next = current.Next
		current.Next = preMiddle.Next
		preMiddle.Next = current
	}

	p1 = head
	p2 = preMiddle.Next
	for p1 != preMiddle{
		preMiddle.Next = p2.Next
		p2.Next = p1.Next
		p1.Next = p2
		p1 = p2.Next
		p2 = preMiddle.Next
	}

}


func insertionSortList(head *ListNode) *ListNode {
	dummy := &ListNode{}
	dummy.Next = head
	prei := dummy
	for i:=dummy.Next; i != nil; {
		prej := dummy
		for j:=dummy.Next; j != i;  {
				fmt.Println("i,j",prei.Val,"-->",i.Val,prej.Val,"-->",j.Val)
				fmt.Printf("%s","before")
				printlink(dummy.Next)
			if j.Val > i.Val{
				prei.Next = j
				prej.Next = i
				i.Next,j.Next = j.Next,i.Next
				i,j = j,i
			}
				fmt.Printf("%s","after")
				printlink(dummy.Next)
			prej = j
			j = j.Next
		}
		prei = i
		i = i.Next
	}

	return dummy.Next
}

func printlink(node *ListNode){
	for node != nil{
		fmt.Printf("%d %s",node.Val,"--> ")
		node = node.Next
	}
	fmt.Println()
}

func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {//|| head.Next.Next == nil{
		return head
	}
	slow,fast := head,head
	var pre *ListNode
	for fast != nil && fast.Next != nil{
		pre = slow
		slow = slow.Next
		fast = fast.Next.Next
	}

	pre.Next = nil

	p1 := sortList(head)
	p2 := sortList(slow)
	return mergesort(p1,p2)
}

func mergesort(p1,p2 *ListNode)*ListNode{
	dummy := ListNode{}
	head := &dummy
	
	for p1 != nil && p2 != nil{
		if p1.Val > p2.Val{
			head.Next = p2
			p2 = p2.Next
		}else{
			head.Next = p1
			p1 = p1.Next
		}
		head = head.Next
	}
	if p1 != nil{
		head.Next = p1
	}
	if p2 != nil{
		head.Next = p2
	}
	return dummy.Next
}

type Point struct {
    X int
    Y int
}

func maxPoints(points []Point) int {
	size := len(points)
	if size <= 0{
		return 0
	}
	xielv := make([][]Point,size)
	for i:=0; i < size; i ++{
		xielv[i] := make([]Point,size)
	}
	for i:=0; i < size; i ++{
		for j:=0; j < i; j ++{
			xielv[i][j] = Point{
				X:points[i].X - points[j].X,
				Y:points[i].Y - points[j].Y
			}
		}
	}
	max := 0
	for i:=0; i < size; i ++{
		for j:=0; j < i; j ++{
			countpoint(i,j,xielv)
		}
	}
	return max
}

func countpoint(p1,p2 int, xie [][]Point){

}

func main() {
	a := &ListNode{Val:2,Next:&ListNode{Val:1}}
	node := &ListNode{Val:3,Next:a}
	node = sortList(node)
	printlink(node)
}

/*
""
[]
*/
