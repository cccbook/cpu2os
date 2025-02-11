import random
import math

class Component:
    def __init__(self, name, width, height, priority=1):
        self.name = name
        self.width = width
        self.height = height
        self.priority = priority
        self.x = 0
        self.y = 0

class ICPlacer:
    def __init__(self, chip_width, chip_height):
        self.chip_width = chip_width
        self.chip_height = chip_height
        self.components = []
        self.placement_grid = None

    def add_component(self, component):
        self.components.append(component)

    def place_components_grid(self):
        """網格佈局策略"""
        # 根據元件數量計算網格佈局
        grid_cols = math.ceil(math.sqrt(len(self.components)))
        grid_rows = math.ceil(len(self.components) / grid_cols)

        cell_width = self.chip_width // grid_cols
        cell_height = self.chip_height // grid_rows

        for i, component in enumerate(sorted(self.components, key=lambda x: x.priority, reverse=True)):
            row = i // grid_cols
            col = i % grid_cols

            component.x = col * cell_width
            component.y = row * cell_height

        return self.components

    def place_components_random(self):
        """隨機佈局策略"""
        for component in self.components:
            component.x = random.randint(0, self.chip_width - component.width)
            component.y = random.randint(0, self.chip_height - component.height)

        return self.components

    def place_components_clustering(self):
        """基於相似性的叢集佈局策略"""
        # 這是一個簡化的叢集策略
        components_by_priority = sorted(self.components, key=lambda x: x.priority, reverse=True)
        
        # 將晶片分成四個象限
        quadrants = [
            (0, 0, self.chip_width//2, self.chip_height//2),
            (self.chip_width//2, 0, self.chip_width, self.chip_height//2),
            (0, self.chip_height//2, self.chip_width//2, self.chip_height),
            (self.chip_width//2, self.chip_height//2, self.chip_width, self.chip_height)
        ]

        for i, component in enumerate(components_by_priority):
            quadrant = quadrants[i % len(quadrants)]
            
            component.x = random.randint(quadrant[0], quadrant[2] - component.width)
            component.y = random.randint(quadrant[1], quadrant[3] - component.height)

        return self.components

    def check_overlap(self):
        """檢查元件是否重疊"""
        for i in range(len(self.components)):
            for j in range(i+1, len(self.components)):
                comp1 = self.components[i]
                comp2 = self.components[j]
                
                if (comp1.x < comp2.x + comp2.width and
                    comp1.x + comp1.width > comp2.x and
                    comp1.y < comp2.y + comp2.height and
                    comp1.y + comp1.height > comp2.y):
                    return True
        return False

    def visualize_placement(self):
        """將佈局以文字形式可視化"""
        print(f"芯片尺寸: {self.chip_width} x {self.chip_height}")
        for component in self.components:
            print(f"元件 {component.name}: 位置 ({component.x}, {component.y}), 尺寸 {component.width} x {component.height}")

# 使用範例
def main():
    # 創建晶片佈局器
    placer = ICPlacer(chip_width=1000, chip_height=800)

    # 添加元件
    placer.add_component(Component("CPU", width=200, height=150, priority=3))
    placer.add_component(Component("Memory", width=150, height=100, priority=2))
    placer.add_component(Component("GPU", width=180, height=130, priority=3))
    placer.add_component(Component("Power Management", width=100, height=80, priority=1))

    # 嘗試不同的佈局策略
    print("網格佈局:")
    placer.place_components_grid()
    placer.visualize_placement()

    print("\n隨機佈局:")
    placer.place_components_random()
    placer.visualize_placement()

    print("\n叢集佈局:")
    placer.place_components_clustering()
    placer.visualize_placement()

    # 檢查是否有重疊
    if placer.check_overlap():
        print("\n警告：元件存在重疊！")
    else:
        print("\n佈局成功，無重疊。")

if __name__ == "__main__":
    main()