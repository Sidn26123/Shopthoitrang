package ptithcm.service;

import java.util.List;

import ptithcm.entity.DonHangEntity;

public interface DonHangService{
	public void luuDonHang(DonHangEntity donHang);
	public void updateDonHang(DonHangEntity donHang);
	public DonHangEntity timDonHangTheoMa(int maDh);
	public List<DonHangEntity> timDonHangCuaUserTheoTrangThai(int maNd, int trangThai);
	
	public List<DonHangEntity> layAllDonHang();	
	public List<DonHangEntity> layDonHangTheoTrangThai(int trangThai);
	public long tinhTongDoanhThuTheoThang(int thang);
	
	public List<String> layMaSanPhamTrongDonHangGanNhatCuaUser(int maNd);
}