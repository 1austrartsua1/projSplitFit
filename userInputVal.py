# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:29:15 2020

Validating User Input
"""

class ExceptLow(Exception):
    pass

class ExceptHigh(Exception):
    pass

def checkUserInput(userInput,rightType,typeAsString,inputName,default=None,low=None,lowAllowed=False,high=None,highAllowed=False):
    try:
        output = rightType(userInput)        
        if low != None:
            if lowAllowed:
                if output<low:
                    raise ExceptLow
            else:
                if output<=low:
                    raise ExceptLow
        
        if high != None:
            if highAllowed:
                if output>high:
                    raise ExceptHigh
            else:
                if output>=high:
                    raise ExceptHigh
        return output                        
    except(ValueError):        
        if default == None:
            raise Exception(f"ERROR: {inputName} is meant to be of type {typeAsString}") 
        else:
            print(f"WARNING: {inputName} is meant to be of type {typeAsString}")
            print(f"Setting it to default {default}")
            return default
    except(ExceptLow):
        if default == None:
            raise Exception(f"ERROR: {inputName} is meant to be greater than {low}") 
        else:
            print(f"WARNING: {inputName} is meant to be greater than {low}")
            print(f"Setting it to default {default}")
            return default
    except(ExceptHigh):
        if default == None:
            raise Exception(f"ERROR: {inputName} is meant to be less than {high}") 
        else:
            print(f"WARNING: {inputName} is meant to be less than {high}")
            print(f"Setting it to default {default}")
            return default
        


