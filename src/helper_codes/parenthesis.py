# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:47:09 2020

@author: rstem
"""
def isParenthesis(c): 
    return ((c == '(') or (c == ')'))  
  
# method returns true if contains valid  
# parenthesis  
def isValidString(str): 
    cnt = 0
    for i in range(len(str)): 
        if (str[i] == '('): 
            cnt += 1
        elif (str[i] == ')'): 
            cnt -= 1
        if (cnt < 0): 
            return False
    return (cnt == 0) 
      
# method to remove invalid parenthesis  
def removeInvalidParenthesis(str): 
    if (len(str) == 0): 
        return
          
    # visit set to ignore already visited  
    visit = set() 
      
    # queue to maintain BFS 
    q = [] 
    temp = 0
    level = 0
      
    # pushing given as starting node into queu 
    q.append(str) 
    visit.add(str) 
    while(len(q)): 
        str = q[0] 
        q.pop() 
        if (isValidString(str)): 
            print(str) 
              
            # If answer is found, make level true  
            # so that valid of only that level  
            # are processed.  
            level = True
        if (level): 
            continue
        for i in range(len(str)): 
            if (not isParenthesis(str[i])): 
                continue
                  
            # Removing parenthesis from str and  
            # pushing into queue,if not visited already  
            temp = str[0:i] + str[i + 1:]  
            if temp not in visit: 
                q.append(temp) 
                visit.add(temp)
                
        return(list(visit))
        
expression = "()())()"
a = removeInvalidParenthesis(expression)


valid = []
for x in a:
    # print(x)
    
    if isValidString(x) == True:
        y = x
        valid.append(y)
        print(x)
        
        
