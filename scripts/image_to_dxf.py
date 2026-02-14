import cv2
import numpy as np
import ezdxf
import argparse
import os

def save_as_svg(contours, width, height, output_path):
    with open(output_path, 'w') as f:
        f.write(f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n')
        for contour in contours:
            path_data = "M "
            for i, point in enumerate(contour):
                x, y = point[0]
                path_data += f"{x},{y} "
                if i == 0:
                    path_data += "L "
            path_data += "Z"
            f.write(f'  <path d="{path_data}" fill="none" stroke="black" stroke-width="1"/>\n')
        f.write('</svg>')

def image_to_dxf(input_path, output_path):
    # 1. 이미지 로드
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 전처리 고도화 (노이즈 제거 및 선 선명화)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    
    # Bilateral Filter는 엣지를 보존하면서 노이즈를 강력하게 제거합니다 (우글거림 방지)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)
    
    # 모폴로지 연산 (끊어진 선 연결 및 노이즈 제거)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 3. 시각적 블루프린트 이미지 생성
    blueprint_viz = np.zeros((height, width, 3), dtype=np.uint8)
    blueprint_viz[:] = (120, 50, 0)
    blueprint_viz[edges > 0] = (255, 255, 255)
    
    viz_output = output_path.replace('.dxf', '_blueprint.png')
    cv2.imwrite(viz_output, blueprint_viz)
    print(f"Success: Blueprint visualization saved to {viz_output}")

    # 4. 윤곽선 추출 및 정형화
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # DXF 설정
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = 4
    msp = doc.modelspace()

    # 직각도 및 직선화를 위한 임계값 (도 단위)
    ANGLE_THRESHOLD = 5.0 

    for contour in contours:
        if cv2.arcLength(contour, True) < 15: # 너무 짧은 선 무시
            continue

        # 단순화 (Douglas-Peucker) - epsilon을 약간 높여 잔떨림 제거
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 2:
            continue

        points = []
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            
            x1, y1 = float(p1[0]), float(height - p1[1])
            x2, y2 = float(p2[0]), float(height - p2[1])

            # 각도 계산 (라디안 -> 도)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # 0, 90, 180, 270도 근처라면 직각 스냅
            # 수평선 스냅 (0, 180, 360도 근처)
            if angle < ANGLE_THRESHOLD or abs(angle - 180) < ANGLE_THRESHOLD or abs(angle - 360) < ANGLE_THRESHOLD:
                y2 = y1 
            # 수직선 스냅 (90, 270도 근처)
            elif abs(angle - 90) < ANGLE_THRESHOLD or abs(angle - 270) < ANGLE_THRESHOLD:
                x2 = x1

            # 첫 번째 점 추가
            if i == 0:
                points.append((x1, y1))
            points.append((x2, y2))

        # 데이터 정리 (중복 점 제거)
        final_points = []
        if points:
            final_points.append(points[0])
            for p in points[1:]:
                if np.linalg.norm(np.array(p) - np.array(final_points[-1])) > 0.5:
                    final_points.append(p)

        if len(final_points) > 2:
            msp.add_lwpolyline(final_points, close=True)

    # 5. SVG 저장 (정형화된 데이터를 기반으로 다시 그리기 위해 points 활용 가능하지만 일단 기존 contour 사용)
    svg_output = output_path.replace('.dxf', '.svg')
    save_as_svg(contours, width, height, svg_output)
    print(f"Success: SVG saved to {svg_output}")

    # 6. DXF 저장
    doc.saveas(output_path)
    print(f"Success: DXF saved to {output_path}")
    
    # 추가로 가공된 이미지(블루프린트 스타일)를 SVG처럼 저장하고 싶다면 vtracer 등을 쓸 수 있지만,
    # 여기서는 OpenCV 윤곽선을 직접 DXF로 변환했습니다.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Image to DXF using OpenCV")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output DXF file")
    
    args = parser.parse_args()
    image_to_dxf(args.input, args.output)
