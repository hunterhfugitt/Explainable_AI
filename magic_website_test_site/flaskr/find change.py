hash_amount = {}

def _find_change_50(total, amount_cards):
    if(total == 0 and amount_cards>-1):
        return 1
    elif(total<0 or amount_cards<0):
        return 0
    else:
        return(_find_change_50(total-50, amount_cards-1) + _find_change_30(total, amount_cards))
    
def _find_change_30(total, amount_cards):
    if(total==0 and amount_cards>-1):
        return 1
    elif(total<0 or amount_cards<0):
        return 0
    else:
        return _find_change_30(total-30, amount_cards-1) + _find_change_10(total, amount_cards)

def _find_change_10(total, amount_cards):
    if(total==0 and amount_cards>-1):
        return 1
    elif(total<0 or amount_cards<0):
        return 0
    else:
        return _find_change_10(total-10, amount_cards-1)
    
def _find_change(total, amount_cards):
    amount1 = 0 
    amount2 = 0
    amount3 = 0
    if(total<30):
        amount3 =  _find_change_10(total,amount_cards)
    elif(total<50):
        amount2 =  _find_change_30(total,amount_cards)
    else:
        amount1 =  _find_change_50(total,amount_cards)
    return(amount1+amount2+amount3)
    
print(_find_change(200,5921))


