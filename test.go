package main

import (
	"fmt"
	"errors"
	"strconv"
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
		fmt.Println("i:",i)
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
						fmt.Println("shift",tdict)
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



func nextPermutation(nums []int)  {
	if len(nums) <= 1{
		return
	}
	
	var i int
	for i=len(nums)-2; i>=0; i--{
		if nums[i] < nums[i+1]{
			break
		}
	}

	var j int
	if i>=0{
		for j=len(nums)-1; j>i ;j--{
			if nums[j] > nums[i]{
				break
			}
		}
		nums[i],nums[j] = nums[j],nums[i]
	}

	//reverse
	for m,n := i+1,len(nums)-1; m<n; m,n = m+1,n-1{
		nums[m],nums[n] = nums[n],nums[m]
	}
}


type Stack []rune

func (s Stack)pop()(Stack,rune,error){
	var r rune
	if len(s) <= 0{
		return s,r,errors.New("empty")
	}
	r = s[len(s) - 1]
	s = s[:len(s)-1]
	return s,r,nil
}

func (s Stack)push(e rune)Stack{
	return append(s,e)
}

func valid(s string)bool{
	var sta Stack
	for _,el := range s{
		if el == '('{
			sta = sta.push(el)
		}else if el == ')'{
			var la rune
			var err error
			sta,la,err = sta.pop()
			if err != nil{
				return false
			}
			if la != '('{
				return false
			}
		}
	}
	if len(sta) != 0{
		return false
	}
	return true
}


func longestValidParentheses1(s string) int {
	max := 0
	for i:=0; i<len(s);i++{
		for j:=i+1; j <= len(s);j++{
			if valid(s[i:j]){
				if j-i > max{
					max = j-i
				}
			}
		}
	}

	return max
}

func longestValidParentheses(s string) int {
	dp := make([]int,len(s))
	max := 0
	for i:=1;i<len(s);i++{
		if s[i] == ')'{
			if s[i-1] == '('{
				if i-2 >= 0{
					dp[i] = dp[i-2]+2
				}else{
					dp[i] = 2
				}
			}else if i-dp[i-1] > 0 && s[i-dp[i-1]-1] == '('{
				if i-dp[i-1]-2 >= 0{
					dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
				}else{
					dp[i] = dp[i-1] + 2
				}
			}
			if dp[i] > max{
				max = dp[i]
			}
		}
	}

	return max
}



func extremeInert(nums []int,target int, left bool)int{
	lo := 0;
	hi := len(nums)
	for lo < hi{
		mid := (lo+hi)/2
		if nums[mid] > target || (left && target==nums[mid]){
			hi = mid 
		}else{
			lo = mid  + 1
		}
	}
		fmt.Println(lo,hi)
	return lo
}



func searchRange(nums []int, target int) []int {
	ret := []int{-1,-1}
	left := extremeInert(nums,target,true)

	if left == len(nums) || nums[left] != target{
		return ret
	}

	ret[0] = left
	ret[1] = extremeInert(nums,target,false)-1
	return ret
}


func newgrid()[][]bool{
	g := make([][]bool,9)
	for i:=0; i < 9; i++{
		g[i] = make([]bool,9)
	}
	return g
}


func isValidSudoku(board [][]byte) bool {
	use1 := newgrid()
	use2 := newgrid()
	use3 := newgrid()
	
	for i:=0; i<len(board); i++{
		for j:=0; j<len(board[i]); j++{
			if board[i][j] == '.'{
				continue
			}
			num := board[i][j] - '0' - 1
			k := i/3*3 + j/3
			if use1[i][num] || use2[j][num] || use3[k][num]{
				return false
			}
			use1[i][num], use2[j][num], use3[k][num] = true,true,true
		}
	}
	return true
}


func validSk(board [][]byte,row,col int,c byte)bool{
	for i:=0; i<9; i++{
		if board[i][col] != '.' && board[i][col] == c{
			return false
		}
		if board[row][i] != '.' && board[row][i] == c{
			return false
		}
		ch := board[row/3*3 + i/3][col/3*3 + i%3]
		if ch != '.' && ch == c{
			return false
		}
	}
	return true
}

func solve(board [][]byte)bool{
	for i:=0 ; i < len(board); i++{
		for j:=0; j<len(board[i]); j++{
			if board[i][j] == '.'{
				for c:='1';c<='9';c++{
					if validSk(board,i,j,byte(c)){
						board[i][j] = byte(c)

						if solve(board){
							return true
						}else{
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


func solveSudoku(board [][]byte)  {
	if len(board) == 0{
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


func sort(a []int){
	for i:=0; i<len(a);i++{
		for j:=i+1; j<len(a);j ++{
			if a[i] > a[j]{
				a[i],a[j] = a[j],a[i]
			}
		}
	}
}

func backtrack(can []int,ret *[][]int,tmp []int,remain,start int){
	if remain < 0{
		return
	}else if remain == 0{
		newtmp := make([]int,0)
		newtmp = append(newtmp,tmp...)
		*ret = append(*ret,newtmp)
	}else{
		for i:=start;i<len(can);i++{
			if i>start && can[i] == can[i-1]{
				continue
			}
			tmp = append(tmp,can[i])
			backtrack(can,ret,tmp,remain-can[i],i+1)
			tmp = tmp[:len(tmp)-1]
		}
	}
}


func combinationSum2(candidates []int, target int) [][]int {
	ret := make([][]int,0)
	tmp := make([]int,0)

	sort(candidates)
	backtrack(candidates,&ret,tmp,target,0)

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
	pre,last := 0,len(nums)-1
	for pre < last{
		if nums[pre] <= 0{
			pre ++
		}else if nums[last] > 0{
			last --
		}else{//nums[pre] > 0 && nums[last] <= 0
			nums[pre],nums[last] = nums[last],nums[pre]
			pre ++
			last --
		}
	}
	fmt.Println(nums)
	//find first positive number
	var i int
	for i=0; i < len(nums); i++{
		if nums[i] > 0{
			break
		}
	}

	for j:=i;j<len(nums);j++{
		if nums[j] != j-i+1{
			return j-i+1
		}
	}
	return len(nums)+1

}

func leftMax(n []int,pos int)int{
	max := 0
	for i:=0; i<pos; i++{
		if n[i] > max{
			max = n[i]
		}
	}
	return max
}

func rightMax(n []int,pos int)int{
	max := 0
	for i:=pos+1; i<len(n); i++{
		if n[i] > max{
			max = n[i]
		}
	}
	return max
}

func min(a,b int)int{
	if a<b{
		return a
	}else{
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

func max(a,b int)int{
	if a>b{
		return a
	}else{
		return b
	}
}

func trap(height []int) int {
	size := len(height)
	left := make([]int,size)
	right := make([]int,size)

	left[0] = height[0]
	for i:=1; i<size; i++{
		left[i] = max(height[i],left[i-1])
	}

	right[size-1] = height[size-1]
	for i:=size-2; i>=0; i--{
		right[i] = max(height[i],right[i+1])
	}
	full := 0
	for i:=0;i < size; i++{
		lr_min := min(left[i],right[i])
		stor := lr_min - height[i]
		if stor > 0{
			full += stor
		}
	}
	return full
}


func multiply(num1 string, num2 string) string {
	s1,s2 := len(num1),len(num2)  
	ans := make([]int,100000)
	var pos int
	for i:=s1-1; i>=0; i--{
		//carry := 0
		c1 := int(num1[i] - '0')
		for j:=s2-1; j>=0; j--{
			c2 := int(num2[j] - '0')
			pos = (s1-i-1) + (s2-j-1)
			ans[pos] = c1 * c2 + ans[pos]
			for ans[pos] > 10{
				carry := ans[pos]/10
				ans[pos] %= 10
				ans[pos+1] += carry
				pos ++
			}
		}
	}
	strans := ""
	for pos >=0 && ans[pos] == 0{
		pos --
	}
	for ;pos >=0 ;pos --{
		strans += strconv.Itoa(ans[pos])
	}
	if strans == ""{
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
    if len(p) == 0{
        return len(s) == 0
    } 
    first_match := len(s) > 0 && ( s[0] == p[0] || p[0] == '.')
    if len(p) >=2 && p[1] == '*'{
        return isMatchRE(s,p[2:]) || (first_match && isMatchRE(s[1:],p))
    }else{
        return first_match && isMatchRE(s[1:],p[1:])
    }
    
}

func jump(nums []int) int {
	if len(nums) < 2{
		return 0
	}
	level := 0
	lnums := []int{0}
	for len(lnums) != 0{
		level ++
		new_lnums := make([]int,0)
		cur_max := 0
		for _,v := range lnums{
			num_v := nums[v]
			if v+num_v > cur_max{
				cur_max = v+num_v
				for i:=1; i<=num_v; i++{
					new_pos := v + i
					if new_pos == len(nums) - 1{
						return level
					}
					if new_pos > len(nums)-1{
						break
					}
					if len(new_lnums) > 0 && new_pos <= new_lnums[len(new_lnums)-1]{
						continue
					}
					new_lnums = append(new_lnums,new_pos)
				}
			}
		}
		lnums = new_lnums
	}
	return 0

}


func permute(nums []int) [][]int {
	ret := make([][]int,0)
	//fmt.Println("passing",nums)
	//defer func(){fmt.Println("return",ret)}()
	if len(nums) == 0{
		return ret
	}
	if len(nums) == 1{
		ret = append(ret,nums)
		return ret
	}
	for i:=0; i < len(nums); i++{
		//fmt.Println(i)
		//fmt.Println(nums[i])
		//fmt.Println("nums",nums)
		new_nums := make([]int,0)
		new_nums  = append(new_nums,nums[:i]...)
		new_nums  = append(new_nums,nums[i+1:]...)
		tmp_ret := permute(new_nums)
		//fmt.Println("after nums",nums)
		for _,v := range tmp_ret{
			tt := []int{nums[i]}
			tt = append(tt,v...)
			ret = append(ret,tt)
		}
	}

	return ret
}

func permuteU(nums []int) [][]int {
    	ret := make([][]int,0)
	if len(nums) == 0{
		return ret
	}
	if len(nums) == 1{
		ret = append(ret,nums)
		return ret
	}
	for i:=0; i < len(nums); i++{
		if i >0 && nums[i] == nums[i-1]{
			continue
		}
		new_nums := make([]int,0)
		new_nums  = append(new_nums,nums[:i]...)
		new_nums  = append(new_nums,nums[i+1:]...)
		tmp_ret := permuteU(new_nums)
		for _,v := range tmp_ret{
			tt := []int{nums[i]}
			tt = append(tt,v...)
			ret = append(ret,tt)
		}
	}

	return ret
}


func permuteUnique(nums []int) [][]int {
	sort(nums)
	return permuteU(nums)
}


func flip(matrix [][]int){
	for i:=0; i<len(matrix)/2; i++{
		matrix[i],matrix[len(matrix)-1-i] = matrix[len(matrix)-1-i],matrix[i]
	}
}


func rotate(matrix [][]int)  {
	//flip first,then     
    flip(matrix)
	for i:=0; i<len(matrix); i++{
		for j:=0; j<len(matrix[i]); j++{
			if i>j{
				matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
			}
		}
	}
}

func validQ(mt [][]bool,row,col int)bool{
    //row
    for i:=0; i<col; i++{
        if mt[row][i]{
            return false
        }
    }
    //line
    for i:=0; i<row; i++{
        if mt[i][col]{
            return false
        }
    }
    //xie
    for i,j := row-1,col-1; i >= 0 && j >= 0; i,j = i-1,j-1{
        if mt[i][j]{
            return false
        }
    }

    for i,j := row-1,col+1; i >=0 && j<len(mt); i,j = i-1,j+1{
        if mt[i][j]{
            return false
        }
    }
    return true
}

func solveNq(ans *[][]string,mt [][]bool,row int){
    if row == len(mt){
        //valid ans, add to ans later
        tmpans := make([]string,0)
        for i:=0; i<len(mt); i++{
            ans_str := ""
            for j:=0; j<len(mt[i]); j++{
                if mt[i][j]{
                    ans_str += "Q"
                }else{
                    ans_str += "."
                }
            }
            tmpans = append(tmpans,ans_str)
        }
        fmt.Println(tmpans)
        *ans = append(*ans,tmpans)
        return
    }
    for i:=0 ; i<len(mt[row]); i++{
        mt[row][i] = true
        if validQ(mt,row,i){
            solveNq(ans,mt,row+1)
        }
        mt[row][i] = false
    }
}

func solveNQueens(n int) [][]string {
    solu := make([][]string,0)
    mt := make([][]bool,n)
    for i:=0 ; i<n ; i ++{
        mt[i] = make([]bool,n)
    }
    solveNq(&solu,mt,0)
    return solu
}

func totalNQueens(n int) int {
    return len(solveNQueens(n))
}



func main() {
	//a := []int{0,1,0,2,1,0,1,3,2,1,2,1}
	//fmt.Println(combinationSum2([]int{10,1,2,7,6,1,5},8))
	//fmt.Println(jump([]int{8,2,4,4,4,9,5,2,5,8,8,0,8,6,9,1,1,6,3,5,1,2,6,6,0,4,8,6,0,3,2,8,7,6,5,1,7,0,3,4,8,3,5,9,0,4,0,1,0,5,9,2,0,7,0,2,1,0,8,2,5,1,2,3,9,7,4,7,0,0,1,8,5,6,7,5,1,9,9,3,5,0,7,5}))
	//fmt.Println(searchRange([]int{1},1))
	fmt.Println(solveNQueens(4))
}


/*

[10,1,2,7,6,1,5]
8
[5,7,7,8,8,10]
8
"wordgoodgoodgoodbestword"
["word","good","best","good"]


"barfoofoobarthefoobarman"
["bar","foo","the"]
*/
