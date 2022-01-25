#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:57:27 2020

@author: tempker
"""

def checkbalance(expression): 
      
    open_tup = tuple('({[') 
    close_tup = tuple(')}]') 
    map = dict(zip(open_tup, close_tup)) 
    queue = [] 
  
    for i in expression: 
        if i in open_tup: 
            queue.append(map[i]) 
        elif i in close_tup: 
            if not queue or i != queue.pop(): 
                try:
                    '['+expresion
                    if i in open_tup:
                        queue.append(map[i])
                    elif i in close_tup:
                        if not queue or i != queue.pop():
                            return "Unbalanced"
                except:
                    pass
                        
                
    if not queue: 
        return "Balanced"
    else:
        try:
            '['+ expresion
            if i in open_tup:
                queue.append(map[i])
            elif i in close_tup:
                if not queue or i != queue.pop():
                    return "Unbalanced"
        except:
            if not queue:
                return "Unbalanced"
            
        return "Unbalanced"
    
def checknobal(expression): 
      
    open_tup = tuple('({[') 
    close_tup = tuple(')}]') 
    map = dict(zip(open_tup, close_tup)) 
    queue = [] 
  
    for i in expression: 
        if i in open_tup: 
            queue.append(map[i]) 
        elif i in close_tup: 
            if not queue or i != queue.pop(): 
                return "Unbalanced"
    if not queue: 
        return "Balanced"
    else: 
        return "Unbalanced"
