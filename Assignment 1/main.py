from binomial_tree import Tree,Node

# Example PUT option
T = Tree(
         current_value=50,
         strike_value=50,
         prop_p=0.5076,
         risk_free_rate=0.10,
         time=5/12,
         volatility=0.40,
         steps=5
        )


T.run('Put')



print(T.tree[(5,4)])