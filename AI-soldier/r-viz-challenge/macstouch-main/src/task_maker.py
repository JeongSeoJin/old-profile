#!/usr/bin/env python
# -- coding: utf-8 --

from copy import deepcopy

class Task:
    def __init__(self):
        self.templete = [{'mode': "init_pos",    'material': -1},
                         {'mode': "tool_get",    'material': -1},
                         {'mode': "vision",      'material': -1},
                         {'mode': "pnp",         'material': -1},
                         {'mode': "tool_return", 'material': -1},
                         {'mode': "finish",      'material': -1}]

    def make_task(self, mode, material):
        task = deepcopy(self.templete[mode])
        task['material'] = material
        # task['material'] = MaterialList[material]
        return task

    def order_to_task(self, order):
        # order.append(1)
        tasks = [self.templete[0]]
        # tasks = []
        

        for material, repeat in enumerate(order):
            if material == 0:
                if repeat:
                    tool_get = self.make_task(1, material)
                    tasks.append(tool_get)

                    pnp = self.make_task(3, material)
                    tasks.append(pnp)

                    tool_return = self.make_task(4, material)
                    tasks.append(tool_return)
                    
            elif material == 1:
                if repeat:
                    tool_get = self.make_task(1, material)
                    tasks.append(tool_get)

                    pnp = self.make_task(3, material)

                    task = [pnp] * repeat
                    
                    tasks.extend(task)

                    tool_return = self.make_task(4, material)
                    tasks.append(tool_return)

            elif material == 2 or material == 5 or material == 6:
                if repeat:
                    pnp = self.make_task(3, material)

                    task = [pnp] * repeat
                    
                    tasks.extend(task)

                    tool_return = self.make_task(4, material)
                    tasks.append(tool_return)

            else:
                if repeat:
                    tool_get = self.make_task(1, material)
                    tasks.append(tool_get)
                    vision = self.make_task(2, material)
                    pnp = self.make_task(3, material)

                    task = [vision, pnp] * repeat
                    
                    tasks.extend(task)

                    tool_return = self.make_task(4, material)
                    tasks.append(tool_return)

            # 좌우 이동 시 꼬임 방지
            if material == 1 or material == 2 or material == 4 or material == 6:
                tasks.append(self.templete[0])

            
        # # bread
        # tool_get = self.make_task(1, 0)
        # tasks.append(tool_get)

        # vision = self.make_task(2, 0)
        # tasks.append(vision)

        # pnp = self.make_task(3, 0)
        # tasks.append(pnp)

        # tool_return = self.make_task(4, 0)
        # tasks.append(tool_return)

        # 좌우 이동 시 꼬임 방지
        tasks.append(self.templete[0])

        finish = self.make_task(5, 0)
        tasks.append(finish)

        return tasks
    

# def main():

#     planner = Task()

#     order = [2, 0, 0, 3, 1, 1, 1, 1]

#     tasks = planner.order_to_task(order)

#     for i in tasks:
#         print(i)

# if __name__ == '__main__':
#     main()