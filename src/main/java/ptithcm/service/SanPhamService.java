package ptithcm.service;

import java.util.List;
import java.util.Map;

import ptithcm.entity.CTDonHangEntity;
import ptithcm.entity.HinhAnhEntity;
import ptithcm.entity.SanPhamEntity;

public interface SanPhamService {
	public SanPhamEntity laySanPham(String maSp);
	public List<SanPhamEntity> laySanPhamTheoMa(String key);
	public List<SanPhamEntity> laySanPhamTheoListMaSP(List<String> listMaSP);
	public List<SanPhamEntity> LaySanPhamMotTrang(int page, int pageSize);
	public List<SanPhamEntity> LaySanPhamMotTran(List<SanPhamEntity> list, int page, int pageSize);
	public List<SanPhamEntity> LaySanPhamMotTrangTheoLoai(String loai, int page, int pageSize);
	public List<SanPhamEntity> layAllSanPham();
	public List<SanPhamEntity> layAllSanPhamDaNgungBan();
	public List<String> laySizeTheoTenSanPham(String maSp);
	public List<Map<String, Integer>> laySizeVaSoLuongTheoMaSP(String maSp);
	public Integer laySoLuongTheoSize(String maSp, String size);
	public List<SanPhamEntity> laySanPhamTheoLoai(String loai);
	public List<SanPhamEntity> laySanPhamTheogioiTinh(String gioiTinh);
	public List<SanPhamEntity> layAllSanPhamTheoLoai(String loai);
	public List<SanPhamEntity> laySanPhamCungLoai(String maSp);
	public List<SanPhamEntity> layTatCaSanPhamCungKieu(String kieu);
	public List<SanPhamEntity> laySanPhamCungKieu(String maSp);
	public List<SanPhamEntity> laySanPhamNgauNhien();
	public List<SanPhamEntity> laySanPhamMoi();
	public List<SanPhamEntity> locSanPhamTrung(List<SanPhamEntity> list);
	public List<SanPhamEntity> locSanPham( List<String> stylesList, int minPrice, int maxPrice);
	public boolean kiemTraSanPhamCoNamTrongGioHang(String maSP);
	public boolean kiemTraSanPhamCoNamTrongDonHang(String maSP);
	public List<CTDonHangEntity> laySanPhamPhoBien(int soLuongSanPham);
	public float tinhSoSaoTB(SanPhamEntity sanPham);
	public void themSanPham(SanPhamEntity sanPham);
	public void updateSanPham(SanPhamEntity sanPham);
	public void xoaSanPham(SanPhamEntity sanPham);
	
	public void themHinhAnhSanPham(HinhAnhEntity hinhAnh);
	public void suaHinhAnhSanPham(HinhAnhEntity hinhAnh);
	public void xoaHinhAnhSanPham(HinhAnhEntity hinhAnh);

}
