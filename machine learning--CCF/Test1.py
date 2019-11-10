# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

        p = [1,2,3]

        a = []
        s = []
        while (p):
            if (p.left != None):
                s.append(p)
                p = p.left
            else:
                a.append(p.val)
                p = s.pop()
                a.append(p.val)
                if (p.right):
                    p = p.right
                    a.append(p.val)
                else:
                    p = s.pop()
        print(a)