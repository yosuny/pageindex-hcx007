from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

def create_text_pdf(filename, text_content):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    # 한글 폰트 설정 (기본 폰트가 한글 미지원이므로 시스템 폰트 사용 시도 또는 대체)
    # 여기서는 간단히 영문으로 테스트하거나, 한글 폰트 경로를 지정해야 함.
    # Mac OS의 경우 AppleGothic 등을 사용할 수 있음.
    try:
        pdfmetrics.registerFont(TTFont('AppleGothic', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'))
        c.setFont('AppleGothic', 10)
    except:
        print("Warning: AppleGothic font not found. Using default font (Korean may not render).")
        c.setFont("Helvetica", 10)

    y = height - 50
    lines = text_content.split('\n')
    
    for line in lines:
        # 긴 줄 처리 (간단히 자르기)
        if len(line) > 80:
            chunks = [line[i:i+80] for i in range(0, len(line), 80)]
            for chunk in chunks:
                c.drawString(50, y, chunk)
                y -= 15
        else:
            c.drawString(50, y, line)
            y -= 15
            
        if y < 50:
            c.showPage()
            try:
                c.setFont('AppleGothic', 10)
            except:
                c.setFont("Helvetica", 10)
            y = height - 50
            
    c.save()
    print(f"Created PDF: {filename}")

# 테스트용 법률 텍스트 (직접 확보한 텍스트로 대체 가능)
text_judgment = """
대법원 2012. 12. 13. 선고 2012다15602 판결 [구상금]

【판시사항】
[1] 사인이 국가의 사무를 처리한 경우 사무관리가 성립하기 위한 요건
[2] 유조선에서 원유가 유출되는 사고 발생 시 해양경찰의 지휘를 받아 방제작업을 보조한 회사가 국가에 방제비용을 청구할 수 있는지 여부(소극)

【판결요지】
[1] 국가 기관이 법령에 따라 처리하여야 할 사무를 사인이 처리한 경우, 그 사무처리가 국가를 위한 것인 동시에 그 사무처리로 국가가 이익을 얻었어야만 국가에 대하여 사무관리에 기한 비용상환을 청구할 수 있다.
[2] 해양오염방지법 등 관련 법령에 의하면, 해양오염사고 발생 시 오염물질을 배출한 선박 소유자 등에게 1차적인 방제의무가 있고, 국가는 선박 소유자 등이 방제의무를 다하지 않거나 긴급한 방제조치가 필요한 경우에 한하여 방제조치를 할 권한과 의무를 가진다. 따라서 민간 방제업체가 해양경찰의 지휘를 받아 방제작업에 참여하였다 하더라도, 이는 선박 소유자 등의 의뢰를 받아 그들의 방제의무를 이행한 것이거나, 영리 목적으로 방제용역을 제공한 것으로 보아야 한다. 따라서 국가를 위한 사무관리라고 볼 수 없어 국가에 방제비용을 청구할 수 없다.

【참조조문】
[1] 민법 제734조, 제739조
[2] 구 해양오염방지법(2007. 1. 19. 법률 제8260호로 개정되기 전의 것) 제48조, 제49조

【주 문】
상고를 기각한다.
상고비용은 원고가 부담한다.
"""

text_terms = """
공정거래위원회 골프장 이용 표준약관 (제10033호)

제1조 (목적)
이 약관은 골프장 사업자(이하 "사업자"라 한다)와 골프장 시설을 이용하는 자(이하 "이용자"라 한다) 사이의 거래 관계를 정하는 것을 목적으로 한다.

제2조 (적용대상)
이 약관은 골프장을 이용하는 모든 내장객에게 적용된다.

제3조 (이용계약의 성립)
이용계약은 이용자가 예약을 하거나 입장절차를 마친 때에 성립한다.

제4조 (예약금)
1. 사업자는 이용예정일로부터 7일 전까지 예약금을 요구할 수 있다.
2. 이용자가 이용예정일 3일 전까지 예약을 취소하는 경우, 사업자는 예약금 전액을 반환하여야 한다.
3. 이용자가 이용예정일 2일 전부터 당일까지 예약을 취소하거나 예약 당일 나타나지 않는 경우, 사업자는 예약금을 반환하지 아니할 수 있다. 단, 천재지변 등 불가항력적인 사유로 인한 경우에는 그러하지 아니하다.

제5조 (이용요금)
1. 골프장 이용요금은 입장료, 카트사용료, 보조원 봉사료 등으로 구분된다.
2. 사업자는 이용요금을 프론트 등 이용자가 보기 쉬운 곳에 게시하여야 한다.

제6조 (물품 구매 강제 금지)
사업자는 이용자에게 식당, 비품 매장 등에서 물품의 구매 또는 이용을 강제하여서는 아니 된다.
"""

# PDF 생성 실행
if __name__ == "__main__":
    base_dir = "comparison/data/documents"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    create_text_pdf(os.path.join(base_dir, "2012da15602.pdf"), text_judgment.strip())
    create_text_pdf(os.path.join(base_dir, "GolfCourse_StandardTerms.pdf"), text_terms.strip())

