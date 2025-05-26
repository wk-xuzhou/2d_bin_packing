"""
二维装箱算法独立脚本

功能：
1. 在给定多边形区域内进行矩形单元的最优排布
2. 支持自定义矩形单元尺寸和间距
3. 通过命令行参数方式提供输入并输出排布结果

使用方法：
python 2d_bin_packing.py --vertices "[[x1,y1],[x2,y2],...]" --length 1.4 --width 2.0 --spacing 0.2

参数说明：
--vertices: 多边形区域顶点坐标列表，格式为[[x1,y1],[x2,y2],...]
--length: 矩形单元长度(米)
--width: 矩形单元宽度(米)
--spacing: 单元间距(米)
"""
import numpy as np
import argparse
from typing import List
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle

class PolygonArea:
    """多边形区域处理类"""
    def __init__(self, vertices: List[List[float]]):
        """初始化多边形区域
        
        Args:
            vertices (List[List[float]]): 多边形顶点坐标列表，每个顶点为[x, y]格式
        """
        self.vertices = np.array(vertices)
        self._calculate_bounding_box()
        self.area = self._calculate_area()
    
    def _calculate_bounding_box(self):
        """计算多边形的边界框"""
        self.min_x = np.min(self.vertices[:, 0])
        self.max_x = np.max(self.vertices[:, 0])
        self.min_y = np.min(self.vertices[:, 1])
        self.max_y = np.max(self.vertices[:, 1])
        
    def _calculate_area(self) -> float:
        """计算多边形面积（使用鞋带公式/高斯面积公式）
        
        Returns:
            float: 多边形面积
        """
        n = len(self.vertices)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        
        return abs(area) / 2.0
        
    def check_rectangle_intersection(self, corners: List[List[float]]) -> bool:
        """检查矩形是否与多边形相交
        
        Args:
            corners (List[List[float]]): 矩形的四个角点坐标
            
        Returns:
            bool: True表示相交，False表示不相交
        """
        # 检查是否至少有一个角点在多边形内
        for corner in corners:
            if self.contains_point(corner):
                return True
                
        # 检查多边形的边是否与矩形的边相交
        rect_edges = [
            [corners[0], corners[1]],  # 底边
            [corners[1], corners[2]],  # 右边
            [corners[2], corners[3]],  # 顶边
            [corners[3], corners[0]]   # 左边
        ]
        
        n = len(self.vertices)
        for i in range(n):
            poly_edge = [self.vertices[i], self.vertices[(i + 1) % n]]
            
            for rect_edge in rect_edges:
                if self._line_segments_intersect(poly_edge[0], poly_edge[1], rect_edge[0], rect_edge[1]):
                    return True
        
        # 检查多边形是否完全包含在矩形内（检查多边形的一个点是否在矩形内）
        if self._point_in_rectangle(self.vertices[0], corners):
            return True
            
        return False
    
    def _line_segments_intersect(self, p1, p2, p3, p4) -> bool:
        """检查两条线段是否相交
        
        Args:
            p1, p2: 第一条线段的端点
            p3, p4: 第二条线段的端点
            
        Returns:
            bool: True表示相交，False表示不相交
        """
        # 计算方向
        d1 = self._direction(p3, p4, p1)
        d2 = self._direction(p3, p4, p2)
        d3 = self._direction(p1, p2, p3)
        d4 = self._direction(p1, p2, p4)
        
        # 检查是否相交
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
            
        # 检查共线情况
        if d1 == 0 and self._on_segment(p3, p4, p1):
            return True
        if d2 == 0 and self._on_segment(p3, p4, p2):
            return True
        if d3 == 0 and self._on_segment(p1, p2, p3):
            return True
        if d4 == 0 and self._on_segment(p1, p2, p4):
            return True
            
        return False
    
    def _direction(self, p1, p2, p3) -> float:
        """计算三点的方向"""
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    def _on_segment(self, p1, p2, p) -> bool:
        """检查点p是否在线段p1p2上"""
        return (min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and 
                min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1]))
                
    def _point_in_rectangle(self, point, corners) -> bool:
        """检查点是否在矩形内"""
        min_x = min(c[0] for c in corners)
        max_x = max(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_y = max(c[1] for c in corners)
        
        return (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y)
    
    def contains_point(self, point: List[float]) -> bool:
        """判断点是否在多边形内（使用射线法）
        
        Args:
            point (List[float]): 待判断的点坐标[x, y]
        
        Returns:
            bool: True表示点在多边形内，False表示不在
        """
        x, y = point
        n = len(self.vertices)
        inside = False
        
        # 修复射线法算法实现
        p1x, p1y = self.vertices[0]
        for i in range(n):
            p2x, p2y = self.vertices[(i + 1) % n]
            if ((p1y <= y and p2y > y) or (p1y > y and p2y <= y)):
                # 计算射线与多边形边的交点x坐标
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    # 只统计射线右侧的交点
                    if x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

class RectangleUnit:
    """矩形单元处理类"""
    def __init__(self, length: float, width: float, spacing: float):
        """初始化矩形单元
        
        Args:
            length (float): 矩形单元长度(米)
            width (float): 矩形单元宽度(米)
            spacing (float): 单元间距(米)
        """
        self.length = length
        self.width = width
        self.spacing = spacing
        self.effective_length = length + spacing
        self.effective_width = width + spacing
    
    def get_corners(self, center_x: float, center_y: float) -> List[List[float]]:
        """根据中心点计算矩形四个角点的坐标
        
        Args:
            center_x (float): 中心点x坐标
            center_y (float): 中心点y坐标
            
        Returns:
            List[List[float]]: 四个角点坐标列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        half_length = self.length / 2
        half_width = self.width / 2
        
        return [
            [center_x - half_length, center_y - half_width],  # 左下
            [center_x + half_length, center_y - half_width],  # 右下
            [center_x + half_length, center_y + half_width],  # 右上
            [center_x - half_length, center_y + half_width]   # 左上
        ]

def calculate_packing(polygon: PolygonArea, rectangle: RectangleUnit, min_corners_inside: int = 2) -> List[List[float]]:
    """计算矩形单元排布
    
    Args:
        polygon (PolygonArea): 多边形区域对象
        rectangle (RectangleUnit): 矩形单元对象
        min_corners_inside (int, optional): 最少需要多少个角点在多边形内，默认为2
    
    Returns:
        List[List[float]]: 排布成功的矩形单元中心坐标列表
    """
    positions = []
    
    # 计算网格步长（考虑单元尺寸和间距）
    step_x = rectangle.effective_length
    step_y = rectangle.effective_width
    
    # 计算边界，增加一定的边界偏移以处理边缘情况
    # 向外扩展边界以确保捕获边缘情况
    start_x = polygon.min_x - step_x/2
    start_y = polygon.min_y - step_y/2
    end_x = polygon.max_x + step_x/2
    end_y = polygon.max_y + step_y/2
    
    # 使用网格方法进行排布
    current_y = start_y
    while current_y <= end_y:
        row_positions = []
        current_x = start_x
        
        while current_x <= end_x:
            # 计算矩形中心点
            center_x = current_x + rectangle.length / 2
            center_y = current_y + rectangle.width / 2
            center = [center_x, center_y]
            
            # 获取矩形的四个角点
            corners = rectangle.get_corners(center_x, center_y)
            
            # 使用改进的相交检测方法
            # 首先检查中心点是否在多边形内
            if polygon.contains_point(center):
                # 然后检查角点
                corners_inside = sum(1 for corner in corners if polygon.contains_point(corner))
                if corners_inside >= min_corners_inside:
                    row_positions.append(center)
            # 即使中心点不在多边形内，也检查矩形是否与多边形相交
            elif polygon.check_rectangle_intersection(corners):
                # 如果相交，计算相交部分的面积比例（简化为角点在内的数量）
                corners_inside = sum(1 for corner in corners if polygon.contains_point(corner))
                if corners_inside >= 1:  # 至少有一个角点在内
                    row_positions.append(center)
            
            current_x += step_x
        
        # 添加当前行的所有位置
        if len(row_positions) > 0:
            positions.extend(row_positions)
        
        current_y += step_y
    
    return positions

def calculate_total_area(positions: List[List[float]], rectangle: RectangleUnit) -> float:
    """计算所有矩形单元的总面积
    
    Args:
        positions (List[List[float]]): 矩形单元中心坐标列表
        rectangle (RectangleUnit): 矩形单元对象
        
    Returns:
        float: 总面积(平方米)
    """
    single_area = rectangle.length * rectangle.width
    return len(positions) * single_area

def visualize_packing(polygon: PolygonArea, rectangle: RectangleUnit, positions: List[List[float]], output_file: str = None):
    """可视化排布结果
    
    Args:
        polygon (PolygonArea): 多边形区域对象
        rectangle (RectangleUnit): 矩形单元对象
        positions (List[List[float]]): 排布成功的矩形单元中心坐标列表
        output_file (str, optional): 输出图片文件路径，不指定则显示图形
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制多边形区域
    poly = MplPolygon(polygon.vertices, alpha=0.3, facecolor='lightblue', edgecolor='blue')
    ax.add_patch(poly)
    
    # 绘制矩形单元
    for pos in positions:
        center_x, center_y = pos
        corners = rectangle.get_corners(center_x, center_y)
        # 计算矩形左下角坐标和宽高
        rect_x = corners[0][0]
        rect_y = corners[0][1]
        rect_width = rectangle.length
        rect_height = rectangle.width
        rect = Rectangle((rect_x, rect_y), rect_width, rect_height, 
                         alpha=0.7, facecolor='lightgreen', edgecolor='green')
        ax.add_patch(rect)
    
    # 设置坐标轴范围，稍微扩大一点以便更好地查看
    margin = max(rectangle.length, rectangle.width) * 2
    ax.set_xlim(polygon.min_x - margin, polygon.max_x + margin)
    ax.set_ylim(polygon.min_y - margin, polygon.max_y + margin)
    
    # 设置标题和标签
    ax.set_title(f'矩形单元排布结果 (共{len(positions)}个单元)')
    ax.set_xlabel('X坐标 (米)')
    ax.set_ylabel('Y坐标 (米)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保持坐标轴比例一致
    ax.set_aspect('equal')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.3, label='多边形区域'),
        Patch(facecolor='lightgreen', edgecolor='green', alpha=0.7, label='矩形单元')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 输出或显示
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_file}")
    else:
        plt.show()

def main():
    """主函数：处理命令行参数并执行排布计算"""
    parser = argparse.ArgumentParser(description='二维装箱算法 - 多边形区域矩形单元最优排布')
    parser.add_argument('--vertices', type=eval, required=True, 
                       help='多边形顶点坐标列表，格式为[[x1,y1],[x2,y2],...]')
    parser.add_argument('--length', type=float, default=1.4, 
                       help='矩形单元长度(米)，默认1.4米')
    parser.add_argument('--width', type=float, default=2.0, 
                       help='矩形单元宽度(米)，默认2.0米')
    parser.add_argument('--spacing', type=float, default=0.2, 
                       help='单元间距(米)，默认0.2米')
    parser.add_argument('--min-corners', type=int, default=2, 
                       help='最少需要多少个角点在多边形内，默认为2')
    parser.add_argument('--output-format', type=str, choices=['simple', 'detailed', 'json'], default='simple',
                       help='输出格式：simple=简单文本, detailed=详细文本, json=JSON格式')
    parser.add_argument('--output-file', type=str, default='',
                       help='输出文件路径，不指定则输出到控制台')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化排布结果')
    parser.add_argument('--visual-output', type=str, default='',
                       help='可视化结果输出图片路径，不指定则显示图形')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式，输出更多信息')
    
    args = parser.parse_args()
    
    # 验证输入参数
    if len(args.vertices) < 3:
        print("错误: 多边形至少需要3个顶点")
        return
        
    if args.length <= 0 or args.width <= 0:
        print("错误: 矩形单元的长度和宽度必须大于0")
        return
        
    if args.spacing < 0:
        print("警告: 单元间距不应小于0，已设置为0")
        args.spacing = 0
    
    # 创建多边形区域和矩形单元对象
    try:
        polygon = PolygonArea(args.vertices)
        rectangle = RectangleUnit(args.length, args.width, args.spacing)
        
        # 计算排布
        positions = calculate_packing(polygon, rectangle, args.min_corners)
    except Exception as e:
        print(f"计算过程中发生错误: {str(e)}")
        return
    
    # 计算统计信息
    total_units = len(positions)
    total_area = calculate_total_area(positions, rectangle)
    polygon_area = polygon.area
    utilization_ratio = (total_area / polygon_area) * 100 if polygon_area > 0 else 0
    
    # 准备输出结果
    result_text = ""
    
    if args.output_format == 'simple':
        result_text += f"成功排布矩形单元数量: {total_units}\n"
        result_text += f"总覆盖面积: {total_area:.2f}平方米 (利用率: {utilization_ratio:.2f}%)\n"
        result_text += "排布位置(中心坐标):\n"
        for pos in positions:
            result_text += f"  {pos}\n"
    
    elif args.output_format == 'detailed':
        result_text += f"多边形区域: {args.vertices}\n"
        result_text += f"多边形面积: {polygon_area:.2f}平方米\n"
        result_text += f"矩形单元尺寸: {args.length}m × {args.width}m ({args.length * args.width:.2f}平方米/个)\n"
        result_text += f"单元间距: {args.spacing}m\n"
        result_text += f"边界框: X[{polygon.min_x:.2f}, {polygon.max_x:.2f}], Y[{polygon.min_y:.2f}, {polygon.max_y:.2f}]\n"
        result_text += f"成功排布矩形单元数量: {total_units}\n"
        result_text += f"总覆盖面积: {total_area:.2f}平方米\n"
        result_text += f"面积利用率: {utilization_ratio:.2f}%\n"
        result_text += "排布位置(中心坐标):\n"
        for i, pos in enumerate(positions):
            result_text += f"  {i+1}: [{pos[0]:.4f}, {pos[1]:.4f}]\n"
    
    elif args.output_format == 'json':
        import json
        result_dict = {
            "polygon": {
                "vertices": args.vertices,
                "area": polygon_area,
                "bounding_box": {
                    "min_x": polygon.min_x, "max_x": polygon.max_x,
                    "min_y": polygon.min_y, "max_y": polygon.max_y
                }
            },
            "rectangle": {
                "length": args.length, 
                "width": args.width, 
                "spacing": args.spacing,
                "area": args.length * args.width
            },
            "result": {
                "count": total_units,
                "total_area": total_area,
                "utilization_ratio": utilization_ratio,
                "positions": positions
            }
        }
        result_text = json.dumps(result_dict, indent=2)
    
    # 输出结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"结果已保存到文件: {args.output_file}")
    else:
        print(result_text)
        
    # 调试信息
    if args.debug:
        print("\n调试信息:")
        print(f"多边形顶点数量: {len(polygon.vertices)}")
        print(f"多边形面积计算值: {polygon.area:.4f}平方米")
        print(f"边界框面积: {(polygon.max_x - polygon.min_x) * (polygon.max_y - polygon.min_y):.4f}平方米")
        print(f"边界框利用率: {(polygon.area / ((polygon.max_x - polygon.min_x) * (polygon.max_y - polygon.min_y))) * 100:.2f}%")
        print(f"理论最大单元数量(忽略形状): {polygon.area / (rectangle.length * rectangle.width):.2f}个")
        print(f"实际排布数量: {total_units}个")
        print(f"排布效率: {(total_units / (polygon.area / (rectangle.length * rectangle.width))) * 100:.2f}%")
    
    # 可视化排布结果
    if args.visualize:
        try:
            visualize_packing(polygon, rectangle, positions, args.visual_output)
        except Exception as e:
            print(f"可视化过程中发生错误: {str(e)}")
            print("提示: 请确保已安装matplotlib库 (pip install matplotlib)")


if __name__ == "__main__":
    main()