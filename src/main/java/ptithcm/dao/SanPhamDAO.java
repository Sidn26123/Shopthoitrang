package ptithcm.dao;

import java.util.List;
import java.util.Map;

import ptithcm.entity.CTDonHangEntity;
import ptithcm.entity.SanPhamEntity;

public interface SanPhamDAO {
	public SanPhamEntity laySanPham(String maSp);
	public List<SanPhamEntity> laySanPhamTheoMa(String key);
	public List<SanPhamEntity> laySanPhamTheoListMaSP(List<String> listMaSP);
	public List<SanPhamEntity> laySanPhamCungTen(String maSp);
	public SanPhamEntity laySanPhamTheoMaVaSize(String maSp, String size);
	public List<Map<String, Integer>> laySizeVaSoLuongTheoMaSP(String maSp);
	public Integer laySoLuongTheoSize(String maSp, String size);
	public List<SanPhamEntity> LaySanPhamMotTrang(int page, int pageSize);
	public List<SanPhamEntity> LaySanPhamMotTrangTheoLoai(String loai, int page, int pageSize);
	public List<SanPhamEntity> layAllSanPham();
	public List<SanPhamEntity> layAllSanPhamDaNgungBan();	
	public List<SanPhamEntity> laySanPhamTheoLoai(String loai);
	public List<SanPhamEntity> laySanPhamTheoGioiTinh(String gioiTinh);
	public List<SanPhamEntity> layAllSanPhamTheoLoai(String loai);
	public List<SanPhamEntity> layTatCaSanPhamCungKieu(String kieu);
	public List<SanPhamEntity> laySanPhamCungKieu(String maSp);
	public List<SanPhamEntity> laySanPhamCungLoai(String maSp);
	public List<SanPhamEntity> laySanPhamNgauNhien();
	public List<SanPhamEntity> laySanPhamMoi();
	public List<SanPhamEntity> locSanPham( List<String> stylesList, int minPrice, int maxPrice);
	public List<CTDonHangEntity> laySanPhamPhoBien(int soLuongSanPham);
	public float tinhSoSaoTB(SanPhamEntity sanPham);
	public void themSanPham(SanPhamEntity sanPham);
	public void updateSanPham(SanPhamEntity sanPham);
	public void xoaSanPham(SanPhamEntity sanPham);
	
	
}
