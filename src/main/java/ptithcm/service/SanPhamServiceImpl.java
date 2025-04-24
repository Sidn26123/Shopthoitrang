package ptithcm.service;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.transaction.Transactional;

import org.hibernate.Query;
import org.hibernate.Session;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ptithcm.entity.CTDonHangEntity;
import ptithcm.entity.GioHangEntity;
import ptithcm.entity.HinhAnhEntity;
import ptithcm.entity.SanPhamEntity;
import ptithcm.dao.CTDonHangDAO;
import ptithcm.dao.HinhAnhDAO;
import ptithcm.dao.SanPhamDAO;

@Service
@Transactional
public class SanPhamServiceImpl implements SanPhamService {
	@Autowired
	SanPhamDAO sanPhamDAO;
	@Autowired
	HinhAnhDAO hinhAnhDAO;
	@Autowired
	gioHangService giohagService;
	@Autowired
	CTDonHangDAO ctdonHangDao;

	public void setSanPhamDAO(SanPhamDAO sanPhamDAO) {
		this.sanPhamDAO = sanPhamDAO;
	}

	@Override
	public SanPhamEntity laySanPham(String maSp) {
		return sanPhamDAO.laySanPham(maSp);
	}

	@Override
	public List<SanPhamEntity> laySanPhamTheoMa(String key) {	    
	    return sanPhamDAO.laySanPhamTheoMa(key);   
	}
	
	@Override
	public List<SanPhamEntity> laySanPhamTheoListMaSP(List<String> listMaSP){
		return sanPhamDAO.laySanPhamTheoListMaSP(listMaSP);
	}
	
	@Override
	public List<SanPhamEntity> LaySanPhamMotTrang(int page, int pageSize){
		return sanPhamDAO.LaySanPhamMotTrang(page, pageSize);
	}
	@Override
	public List<SanPhamEntity> LaySanPhamMotTran(List<SanPhamEntity> list, int page, int pageSize){		
		List<SanPhamEntity> SPMotTrang = new ArrayList<>();
		int offset = page * pageSize;
		int endIndex = pageSize + offset;
		if (endIndex>list.size()) endIndex = list.size();
		for (int i =offset; i<endIndex;i++) {
			SPMotTrang.add(list.get(i));
		}
		return SPMotTrang;
	}
	@Override
	public List<SanPhamEntity> LaySanPhamMotTrangTheoLoai(String loai, int page, int pageSize){
		return sanPhamDAO.LaySanPhamMotTrangTheoLoai(loai, page, pageSize);
	}
	@Override
	public List<String> laySizeTheoTenSanPham(String maSp){
		SanPhamEntity sp= sanPhamDAO.laySanPham(maSp);
		List<String> sizes = new ArrayList<>();
		List<SanPhamEntity> listSP = sanPhamDAO.laySanPhamCungTen(maSp);
	    String spSizes = sp.getSize().trim();
	    String maSpGoc = maSp.substring(0, maSp.length() - spSizes.length()); // Loại bỏ 2 ký tự cuối của maSp

	    for (SanPhamEntity sp2 : listSP) {
	        String productMa = sp2.getMaSP();
	        String productSizes = sp2.getSize().trim();

	        // Kiểm tra 
	        if (productMa.substring(0, productMa.length() - productSizes.length()).equals(maSpGoc)) {
	            sizes.add(productSizes);
	        }
	    }
	    return sizes;
	}
	
	@Override
	public List<SanPhamEntity> laySanPhamTheoLoai(String loai) {
		return sanPhamDAO.laySanPhamTheoLoai(loai);
	}
	
	@Override
	public List<SanPhamEntity> layAllSanPhamTheoLoai(String loai) {
		return sanPhamDAO.layAllSanPhamTheoLoai(loai);
	}

	@Override
	public List<SanPhamEntity> laySanPhamCungLoai(String maSp) {
		return sanPhamDAO.laySanPhamCungLoai(maSp);
	}
	
	@Override
	public List<SanPhamEntity> layTatCaSanPhamCungKieu(String kieu){
		return sanPhamDAO.layTatCaSanPhamCungKieu(kieu);
	}
	@Override
	public List<SanPhamEntity> laySanPhamCungKieu(String maSp) {
		return sanPhamDAO.laySanPhamCungKieu(maSp);
	}
	@Override
	public List<Map<String, Integer>> laySizeVaSoLuongTheoMaSP(String maSp){
		return sanPhamDAO.laySizeVaSoLuongTheoMaSP(maSp);
	}
	@Override
	public List<SanPhamEntity> laySanPhamNgauNhien() {
		return sanPhamDAO.laySanPhamNgauNhien();
	}
	
	@Override
	public List<SanPhamEntity> laySanPhamMoi() {
		return sanPhamDAO.laySanPhamMoi();
	}
	
	@Override
	public List<SanPhamEntity> locSanPhamTrung(List<SanPhamEntity> list) {
		Set<String> uniqueSet = new HashSet<>();
		List<SanPhamEntity> result = new ArrayList<>();
		for (SanPhamEntity sanPham : list) {

			if (!uniqueSet.contains(sanPham.getTenSanPham())) {
				result.add(sanPham);
				uniqueSet.add(sanPham.getTenSanPham());
			}
		}
		return result;
	}
	
	@Override
	public List<SanPhamEntity> locSanPham( List<String> stylesList, int minPrice, int maxPrice){
		return sanPhamDAO.locSanPham(stylesList , minPrice, maxPrice);
	}
	
	@Override
	public boolean kiemTraSanPhamCoNamTrongGioHang(String maSP) {
		List<GioHangEntity> danhSachGioHang = giohagService.layAllGioHang();
	    
	    for (GioHangEntity gioHang : danhSachGioHang) {
	        if (gioHang.getSanPham().getMaSP().equals(maSP)) {
	            return true; // có nằm trong giỏ hàng
	        }
	    }
	    
	    return false; // Không có nằm trong giỏ hàng
	}
	
	@Override
	public boolean kiemTraSanPhamCoNamTrongDonHang(String maSP) {
		List<CTDonHangEntity> danhSachCTDonHang = ctdonHangDao.layAllCTDonHang();
	    
	    for (CTDonHangEntity ctdonHang : danhSachCTDonHang) {
	        if (ctdonHang.getSanPham().getMaSP().equals(maSP)) {
	            return true; // có nằm trong đơn hàng
	        }
	    }
	    
	    return false; // Không có nằm trong đơn hàng
	}

	@Override
	public float tinhSoSaoTB(SanPhamEntity sanPham) {
		return sanPhamDAO.tinhSoSaoTB(sanPham);
	}

	@Override
	public List<SanPhamEntity> layAllSanPham() {
		return sanPhamDAO.layAllSanPham();
	}

	@Override
	public List<SanPhamEntity> layAllSanPhamDaNgungBan() {
		return sanPhamDAO.layAllSanPhamDaNgungBan();
	}

	@Override
	public void themSanPham(SanPhamEntity sanPham) {
		sanPhamDAO.themSanPham(sanPham);
	}
	
	@Override
	public void updateSanPham(SanPhamEntity sanPham) {
	    sanPhamDAO.updateSanPham(sanPham);
		
	}

	@Override
	public void xoaSanPham(SanPhamEntity sanPham) {
		sanPhamDAO.xoaSanPham(sanPham);
	}

	@Override
	public void themHinhAnhSanPham(HinhAnhEntity hinhAnh) {
		hinhAnhDAO.themHinhAnhSanPham(hinhAnh);
		
	}

	@Override
	public void suaHinhAnhSanPham(HinhAnhEntity hinhAnh) {
		hinhAnhDAO.suaHinhAnhSanPham(hinhAnh);
		
	}

	@Override
	public void xoaHinhAnhSanPham(HinhAnhEntity hinhAnh) {
		hinhAnhDAO.xoaHinhAnhSanPham(hinhAnh);
		
	}

	@Override
	public List<SanPhamEntity> laySanPhamTheogioiTinh(String gioiTinh) {
		return sanPhamDAO.laySanPhamTheoGioiTinh(gioiTinh);
	}
	

	@Override
	public List<CTDonHangEntity> laySanPhamPhoBien(int soLuongSanPham){
		return sanPhamDAO.laySanPhamPhoBien(soLuongSanPham);
	}
	
	@Override
	public Integer laySoLuongTheoSize(String maSp, String size) {
		return sanPhamDAO.laySoLuongTheoSize(maSp, size);
	}
	
}
