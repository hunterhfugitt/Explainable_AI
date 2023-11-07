class Solution(object):
    def subarraySum(self, nums, k):
        sum_hash = {}
        amount = 0
        start = 0
        total = 0
        previous = 0
        previously_used = 0
        key = {}
        key_backward = {}
        exists = {}
        amount_key = {}
        previos_key ={}
        
        for x in range (0,len(nums)):
            key[x] = k-nums[x]
            total = total + nums[x]
            sum_hash[x] = total
            if(key[x] == 0):
                amount = amount + 1
        for x in range (0,len(nums)):
            key[x] = k-nums[x]
            total = total + nums[x]
            sum_hash[x] = total
            if(key[x] == 0):
                amount = amount + 1
            # if(x!=0):
            #     if key[x] == sum_hash[x-1]:
            #         amount = amount + 1
            #     else:
            #         key[x] = key[x] - sum_hash[x-1]
        total = 0
        for x in range (0,len(nums)):
            key_backward[len(nums)-1-x] = k-nums[len(nums)-1-x]
            total = total + nums[len(nums)-1-x]
            sum_hash[len(nums)-1-x] = total
            if(key_backward[len(nums)-1-x] == 0):
                amount = amount + 1
            if(x!=0):
                if key_backward[len(nums)-1-x] == sum_hash[len(nums)-x]:
                    amount = amount + 1
                else:
                    key_backward[len(nums)-1-x] = key_backward[len(nums)-1-x] - sum_hash[len(nums)-x]    
                
                
        # for x in range (0,len(nums)):
        #     total = total + nums[x]
        #     sum_hash[x] = total
        #     key[x] = k-nums[x]
        #     if(x!=0):
        #         sum_hash[f"{x}+{x-1}"] = total - sum_hash[x-1]
        # for x in range (0,len(nums)):
        #     total = total + nums[x]
        #     sum_hash[x] = total
             
        # for x in range(0,len(nums)):
        #     if(sum_hash[x] == k):
        #         amount = amount + 1
        #     if(x!=0):
        #         if(sum_hash[x]-sum_hash[x-1] == k):
        #             amount = amount + 1
        #         if(sum_hash[x+1] == k):
        #             amount = amount + 1
            
                
        # return(amount)
                