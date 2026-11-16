"""Recompose the existing vector overview; preserve robot images; match panel sizes and title typography."""
from pathlib import Path
from io import BytesIO
import json,hashlib
import pymupdf as f
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, black
P=Path(__file__).resolve().parents[1];src=P/'archive/20261101_figures_references_windows/figures/fig1_concept.pdf';orig=f.open(src)
W=1600.;H=1050.;DY=70.;panel_height=525.;bottom_clip=f.Rect(0,2290,5849,4243.8203125)
d=f.open();pg=d.new_page(width=W,height=H)
# Preserve the annotation and training photos as vector/bitmap PDF content, scaled uniformly.
pg.show_pdf_page(f.Rect(0,DY,380,422+DY),orig,0,clip=f.Rect(0,0,1690,1875))
pg.show_pdf_page(f.Rect(390,DY,941,429+DY),orig,0,clip=f.Rect(1700,0,4065,1840))
# Preserve the six photographs and sequence arrow, using uniform scaling.
bottom_width=1320.;bottom_top=590.;bottom_height=bottom_clip.height*bottom_width/bottom_clip.width
pg.show_pdf_page(f.Rect(140,bottom_top,1460,bottom_top+bottom_height),orig,0,clip=bottom_clip)
buf=BytesIO();c=canvas.Canvas(buf,pagesize=(W,H))
def rect(x,y,w,h):c.rect(x,H-y-h,w,h,stroke=1,fill=0)
def line(x1,y1,x2,y2):c.line(x1,H-y1,x2,H-y2)
def text(x,y,s,size=23,font='Helvetica',color=black):
 c.setFillColor(color);c.setFont(font,size);c.drawString(x,H-y,s);c.setFillColor(black)
def centered(x,y,s,size=23,font='Helvetica'):
 c.setFont(font,size);c.drawCentredString(x,H-y,s)
c.saveState();c.translate(0,-DY)
# Remove the old small (a) label; both panel labels are drawn together below.
c.setFillColorRGB(1,1,1);c.rect(0,H-48,64,48,stroke=0,fill=1);c.setFillColor(black)
# Identify measured power separately from predictive body-state inputs.
c.setFillColorRGB(1,1,1);c.rect(190,H-374,188,74,stroke=0,fill=1);c.setFillColor(black)
text(201,324,'Supply power',22);text(201,351,'(training target)',20)
# Ridge-regression arrow.
c.setLineWidth(2.3);line(950,243,1078,243);line(1078,243,1068,234);line(1078,243,1068,252)
centered(1014,201,'Ridge',24);centered(1014,229,'regression',24)
# Explicit model ladder requested by the author.
x=1092.;w=502.;c.setLineWidth(2.5);rect(x,13,w,411)
centered(x+w/2,56,'Power prediction models',29,'Helvetica-Bold')
c.setStrokeColor(HexColor('#A7A7A7'));c.setLineWidth(.9)
for y in [78,155,233,327]:line(x+15,y,x+w-15,y)
c.setStrokeColor(black)
red=HexColor('#B42222')
text(x+23,124,'R1',28,'Helvetica-Bold');text(x+96,123,'Joint speed + tracking error',23)
text(x+23,202,'R2',28,'Helvetica-Bold');text(x+96,202,'R1',23)
text(x+96+c.stringWidth('R1','Helvetica',23),202,' + joint-pair features',23,'Helvetica',red)
text(x+23,283,'R3',28,'Helvetica-Bold');text(x+96,270,'R2',23)
text(x+96+c.stringWidth('R2','Helvetica',23),270,' + body inclination',23,'Helvetica',red)
text(x+96,301,'+ assigned support labels',23,'Helvetica',red)
c.restoreState()
c.setLineWidth(1.4);line(0,panel_height,W,panel_height)
text(12,38,'(a)',32);text(85,38,'Learning power prediction models from simple experiences',32)
text(12,563,'(b)',32);text(85,563,'Predicting energy for unseen whole-body motions',32)
c.save();overlay=f.open(stream=buf.getvalue(),filetype='pdf');pg.show_pdf_page(pg.rect,overlay,0)
eq=f.open(P/'qa/20261101_revision/fig1_equation.pdf');pg.show_pdf_page(f.Rect(x+55,343+DY,x+w-55,410+DY),eq,0)
out=P/'figures/fig1_concept.pdf';d.save(out,garbage=4,deflate=True)
pg.get_pixmap(matrix=f.Matrix(1,1),alpha=False).save(P/'qa/20261101_revision/Fig1_updated.png')
(P/'qa/20261101_revision/fig1_verification.json').write_text(json.dumps(dict(original_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),original_bottom_clip=list(bottom_clip),bottom_photos_and_sequence_arrow_preserved=True,panel_heights=[525,525],panel_label_and_title_font="Helvetica 32",robot_images_not_regenerated=True,notes='Vector recomposition of original PDF; measured power marked as training target, support labels assigned, three explanatory labels removed; preceding-model names in black, added feature descriptions in red; equal panel heights, matched panel labels, and titles for both panels.'),indent=2)+'\n')
print(out)
