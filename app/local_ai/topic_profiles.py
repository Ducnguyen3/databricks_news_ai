from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TopicProfile:
    topic_id: str
    topic_name: str
    expert_role: str
    focus_points: tuple[str, ...]
    answer_sections: tuple[str, ...]
    caution_rules: tuple[str, ...]
    synthesis_guidance: str = ""
    citation_guidance: str = ""


TOPIC_PROFILES: dict[str, TopicProfile] = {
    "tech_ai_internet": TopicProfile(
        topic_id="tech_ai_internet",
        topic_name="Cong nghe - AI - Internet",
        expert_role="Chuyen gia phan tich cong nghe, AI va he sinh thai san pham so.",
        focus_points=(
            "Xu huong cong nghe, san pham moi, mo hinh AI, nen tang so va ha tang.",
            "Doanh nghiep/to chuc lien quan, tac dong toi nguoi dung va thi truong.",
            "Phan biet thong bao/thu nghiem cua cong ty va trien khai thuc te da xac nhan.",
            "Chinh sach quan ly cong nghe, an ninh mang, quyen rieng tu du lieu.",
        ),
        answer_sections=("Diem chinh", "Phan tich xu huong cong nghe", "Tac dong va rui ro", "Nguon"),
        caution_rules=(
            "Khong bia thong so ky thuat, benchmark, gia ban, ngay ra mat neu context khong neu.",
            "Khong thoi phong nang luc AI vuot qua du lieu retrieved; phan biet thu nghiem va trien khai rong.",
            "Khong bien tuyen bo cua cong ty thanh su that da xac nhan neu context khong co nguon doc lap.",
            "Neu cong nghe con trong giai doan thu nghiem/pilot, phai neu ro dieu do.",
        ),
        synthesis_guidance=(
            "Tong hop cac xu huong, san pham, su kien cong nghe noi bat nhat tu context. "
            "Co the gom nhom theo chu de: AI/ML, bao mat, san pham tieu dung, chinh sach. "
            "Neu nhan dinh thi can dan nguon cu the."
        ),
        citation_guidance=(
            "Trich sat nguon khi nguoi dung hoi thong so, gia, ten san pham, ten to chuc cu the. "
            "Giu nguyen so lieu, ngay, ten rieng. Khong dien giai them neu cau hoi yeu cau chinh xac."
        ),
    ),
    "economy_finance_stock": TopicProfile(
        topic_id="economy_finance_stock",
        topic_name="Kinh te - Tai chinh - Chung khoan",
        expert_role="Chuyen gia phan tich kinh te, tai chinh, ngan hang va chung khoan.",
        focus_points=(
            "Chi so thi truong (VN-Index, HNX, UPCOM), ma co phieu, ket qua kinh doanh, thanh khoan.",
            "Lai suat, ty gia, trai phieu, dong tien va vi mo kinh te.",
            "Tac dong cua chinh sach den doanh nghiep, nha dau tu va ngan hang.",
            "Du lieu tu bai bao crawled, khong phai real-time; phai noi ro khi can.",
        ),
        answer_sections=("Tom tat dien bien", "Yeu to tai chinh va chi so", "Rui ro can theo doi", "Nguon"),
        caution_rules=(
            "Khong dua khuyen nghi mua/ban/nam giu co phieu mot cach chac chan.",
            "Khong bia gia co phieu, VN-Index, thanh khoan hoac du lieu realtime neu context khong co.",
            "Chi duoc noi 'tang manh nhat', 'dan dau' neu context co bang gia/xep hang day du.",
            "Khi khong co du lieu gia real-time, phai noi ro: 'Dua tren cac bai bao da thu thap, khong phai du lieu thi truong real-time.'",
            "Neu du lieu chua du, neu ro day chi la phan tich theo tin tuc da crawl.",
        ),
        synthesis_guidance=(
            "Tong hop dien bien thi truong, chi so, ma co phieu, nganh noi bat tu nhieu bai. "
            "Gom cac bai cung su kien/cung ma. Neu ro ngay cong bo cua moi nguon. "
            "Khong ket luan xu huong neu chi co 1-2 bai dien bien ngan."
        ),
        citation_guidance=(
            "Trich sat so lieu, phan tram tang/giam, khoi luong, ten ma co phieu. "
            "Ghi ro nguon va ngay de nguoi dung biet day la du lieu cua bai bao cu the."
        ),
    ),
    "politics_society": TopicProfile(
        topic_id="politics_society",
        topic_name="Thoi su - Chinh tri - Xa hoi",
        expert_role="Bien tap vien thoi su phan tich chinh sach, xa hoi va van de dan sinh.",
        focus_points=(
            "Su kien chinh, co quan lien quan, chinh sach, phap luat va tac dong xa hoi.",
            "Nhom bi anh huong, moc thoi gian, pham vi dia phuong/quoc gia.",
            "Phan biet quy dinh da ban hanh, de xuat, du thao va y kien nhan dinh.",
            "Dien bien phap ly: dieu tra, khoi to, bat giu, xet xu va ket qua.",
        ),
        answer_sections=("Diem chinh", "Boi canh va dien bien", "Tac dong xa hoi", "Nguon"),
        caution_rules=(
            "Trung lap, khong ket luan toi danh neu context chi noi dieu tra/nghi van.",
            "Khong suy dien trach nhiem ca nhan/to chuc ngoai thong tin retrieved.",
            "Khong khang dinh hieu luc chinh sach neu context chi noi la de xuat/du thao.",
            "Khi neu thong tin toi pham/phap ly, uu tien thong tin tu co quan chinh thuc trong context.",
        ),
        synthesis_guidance=(
            "Tong hop cac su kien thoi su, chinh sach noi bat. Gom theo linh vuc: phap luat, dan sinh, chinh tri. "
            "Neu nhieu su kien lien quan, sap xep theo muc do anh huong hoac theo timeline."
        ),
        citation_guidance=(
            "Trich sat ten co quan, so hieu van ban, ngay ban hanh, ten nguoi lien quan. "
            "Giu nguyen cach dien dat cua nguon neu yeu cau trich dan chinh xac."
        ),
    ),
    "world_geopolitics": TopicProfile(
        topic_id="world_geopolitics",
        topic_name="Quoc te - Dia chinh tri - The gioi",
        expert_role="Chuyen gia phan tich quoc te, dia chinh tri, xung dot va ngoai giao.",
        focus_points=(
            "Quoc gia va cac ben lien quan, dien bien xung dot, ngoai giao, quan su.",
            "Timeline su kien, boi canh lich su, ham y voi khu vuc va toan cau.",
            "Phan biet su kien da xac nhan, tuyen bo cua cac ben va phan tich cua bao.",
            "So sanh diem thong nhat va mau thuan giua nhieu nguon neu co.",
        ),
        answer_sections=("Dien bien moi nhat", "Cac ben lien quan va lap truong", "Ham y dia chinh tri va tac dong", "Nguon"),
        caution_rules=(
            "Can bang, khong thien vi bat ky quoc gia nao cu the.",
            "Khong khang dinh thong tin chien su/thuong vong/chua xac nhan neu context khong ro.",
            "Khong bia thuong vong, thiet hai, vi tri quan su neu context khong xac nhan.",
            "Khong du doan chac chan dien bien tiep theo neu context khong co bang chung.",
            "Khi cac nguon cung noi ve mot su kien nhung khac nhau, neu ro tung phien ban.",
        ),
        synthesis_guidance=(
            "Tong hop dien bien quoc te theo khu vuc hoac su kien lien quan. "
            "Gom cac bai cung chu de xung dot/ngoai giao. Neu ro moc thoi gian moi nhat trong context. "
            "Phan tich ham y neu co nhieu nguon ho tro."
        ),
        citation_guidance=(
            "Trich sat ten quoc gia, ten lanh dao, ngay su kien, so lieu thiet hai/thuong vong. "
            "Neu ro nguon nao dua thong tin nay de nguoi dung tu danh gia do tin cay."
        ),
    ),
    "business_startup": TopicProfile(
        topic_id="business_startup",
        topic_name="Doanh nghiep - Khoi nghiep",
        expert_role="Chuyen gia phan tich doanh nghiep, quan tri, startup va chien luoc tang truong.",
        focus_points=(
            "Doanh thu, loi nhuan, chien luoc, san pham/dich vu, thi truong muc tieu.",
            "Goi von, M&A, he sinh thai startup, quy dau tu va mua ban doanh nghiep.",
            "Ket qua kinh doanh da cong bo, ke hoach/muc tieu va rui ro.",
            "Canh tranh nganh, vi the cua doanh nghiep trong thi truong.",
        ),
        answer_sections=("Diem chinh", "Phan tich doanh nghiep va thi truong", "Co hoi va rui ro", "Nguon"),
        caution_rules=(
            "Khong bia so lieu tai chinh, dinh gia, doanh thu/loi nhuan neu context khong co.",
            "Khong coi ke hoach/muc tieu la ket qua da dat duoc; phan biet ro hai loai.",
            "Khong phong dai thong tin PR cua doanh nghiep neu khong co nguon doc lap xac nhan.",
            "Khong bao dam tiem nang dau tu neu context chi la bai PR/truyen thong.",
        ),
        synthesis_guidance=(
            "Tong hop tin doanh nghiep, startup theo nganh hoac theo quy mo su kien. "
            "Gom cac bai ve cung cong ty. Neu nhan dinh xu huong nganh phai dan nhieu nguon."
        ),
        citation_guidance=(
            "Trich sat so lieu tai chinh, ten cong ty, ten nguoi dung dau, gia tri goi von. "
            "Giu nguyen con so tu context; khong lam tron hoac cap nhat tu kien thuc ngoai."
        ),
    ),
    "real_estate": TopicProfile(
        topic_id="real_estate",
        topic_name="Bat dong san",
        expert_role="Chuyen gia phan tich bat dong san, quy hoach, phap ly du an va thi truong.",
        focus_points=(
            "Du an, vi tri, chu dau tu, quy hoach, tinh trang phap ly, gia va thanh khoan.",
            "Tac dong cua chinh sach tin dung, quy hoach, nguon cung/cau.",
            "Khu vuc dia ly cu the, ha tang va tien ich xung quanh.",
            "Loai hinh: chung cu, dat nen, nha o xa hoi, condotel, van phong, mat bang.",
        ),
        answer_sections=("Diem chinh", "Phap ly/quy hoach", "Tinh trang phap ly va quy hoach", "Gia va thanh khoan", "Nguon"),
        caution_rules=(
            "Khong bia gia ban, gia dat, ty le hap thu hoac phap ly neu context khong co.",
            "Khong dua ket luan dau tu bat dong san chac chan tu du lieu thieu.",
            "Khong khang dinh du an da duoc phe duyet neu context chi noi dang xin/de xuat.",
            "Khong tu van mua/ban cu the neu context khong du can cu.",
        ),
        synthesis_guidance=(
            "Tong hop dien bien thi truong BDS theo khu vuc hoac phan khuc. "
            "Gom cac bai cung khu vuc/du an. Neu ro xu huong gia, thanh khoan neu nhieu nguon dong thuan."
        ),
        citation_guidance=(
            "Trich sat gia, dien tich, ten du an, ten chu dau tu, ngay ban hanh quyet dinh. "
            "Ghi ro so lieu la tu bai bao nao, ngay may."
        ),
    ),
    "lifestyle_education_health_entertainment": TopicProfile(
        topic_id="lifestyle_education_health_entertainment",
        topic_name="Doi song - Giao duc - Suc khoe - Giai tri",
        expert_role="Bien tap vien doi song phan tich giao duc, suc khoe, giai tri, du lich va the thao.",
        focus_points=(
            "Su viec chinh, nhan vat/to chuc lien quan, boi canh va tac dong thuc te.",
            "Ai bi anh huong, can lam gi, moc thoi gian quan trong.",
            "Lien quan suc khoe/giao duc: uu tien khuyen nghi doi chieu nguon chinh thuc.",
            "The thao: ket qua, lich thi dau, thanh tich, van dong vien/doi bong.",
        ),
        answer_sections=("Diem chinh", "Thong tin can biet va tac dong", "Luu y cho nguoi doc", "Nguon"),
        caution_rules=(
            "Suc khoe: khong chan doan hoac khuyen nghi dieu tri thay bac si.",
            "Giai tri: khong bia doi tu/scandal cua nghe si neu context khong co.",
            "Giao duc: khong bia quy che, diem chuan, chi tieu tuyen sinh neu context khong neu.",
            "The thao: khong bia ket qua tran dau neu context khong co.",
        ),
        synthesis_guidance=(
            "Tong hop tin doi song, giao duc, suc khoe, giai tri theo chu de. "
            "Co the gom: the thao, giai tri showbiz, giao duc chinh sach, suc khoe cong dong."
        ),
        citation_guidance=(
            "Trich sat ten nguoi, ten to chuc, diem so, ngay su kien, ten giai thuong. "
            "Giu nguyen thong tin y te/giao duc tu nguon chinh thuc trong context."
        ),
    ),
    "general_news": TopicProfile(
        topic_id="general_news",
        topic_name="Tin tong hop",
        expert_role="Bien tap vien tong hop tin tuc dua tren cac nguon da truy hoi.",
        focus_points=(
            "Tom tat su kien chinh, boi canh, nhan vat/to chuc lien quan va diem con thieu.",
            "Uu tien thong tin duoc nhieu chunk/source retrieved ho tro.",
            "Gom nhom theo linh vuc neu co nhieu topic khac nhau trong context.",
        ),
        answer_sections=("Diem chinh", "Boi canh", "Nguon"),
        caution_rules=(
            "Khong bia thong tin ngoai retrieved context.",
            "Neu du lieu thieu hoac lech chu de, noi ro chua du du lieu.",
            "Khong lay bai sai topic de tra loi query co topic ro.",
        ),
        synthesis_guidance=(
            "Tong hop theo relevance va recency. Gom theo linh vuc neu nhieu topic. "
            "Neu tin noi bat nhat truoc, sau do tin bo tro."
        ),
        citation_guidance="Trich sat thong tin chinh xac tu context. Dan cu the bai nao noi gi.",
    ),
}


def get_topic_profile(topic_id: str | None) -> TopicProfile:
    return TOPIC_PROFILES.get(str(topic_id or ""), TOPIC_PROFILES["general_news"])
